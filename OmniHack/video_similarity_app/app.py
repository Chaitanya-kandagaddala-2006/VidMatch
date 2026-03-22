import os
import subprocess
import glob
import shutil
import concurrent.futures
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-prod")
app.config["UPLOAD_FOLDER"] = os.getenv("UPLOAD_FOLDER", "uploads")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", 104857600))

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "mp3", "wav", "m4a", "flac"}

# BLIP model — lazy loaded once, reused for all requests
_blip_processor = None
_blip_model = None
_device = None

def load_blip():
    global _blip_processor, _blip_model, _device
    if _blip_processor is None:
        print("Loading BLIP Large model for better accuracy (first time ~1 min, downloads ~1.9GB)...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {_device}")
        
        # FIX: use_fast=True suppresses the slow processor warning
        _blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            use_fast=True
        )
        
        # Optimize by using float16 if on GPU 
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            torch_dtype=dtype
        ).to(_device)
        
        _blip_model.eval()
        print("BLIP Large model loaded and optimized.")
    return _blip_processor, _blip_model, _device


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def extract_audio_from_video(video_path):
    """Use ffmpeg to rip audio from video into a WAV file"""
    audio_path = video_path.rsplit(".", 1)[0] + "_audio.wav"
    result = subprocess.run(
        ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1",
         "-f", "wav", audio_path, "-y"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0 and os.path.exists(audio_path):
        return audio_path
    raise Exception(f"ffmpeg audio extraction failed: {result.stderr[:300]}")


def transcribe_audio(audio_path):
    """
    Google Speech Recognition (free).
    Returns (transcript_str, speech_found_bool)
    Handles long audio by splitting into 55-second chunks.
    """
    recognizer = sr.Recognizer()
    import wave

    try:
        with wave.open(audio_path, "r") as wf:
            duration = wf.getnframes() / float(wf.getframerate())
    except Exception:
        duration = 30.0

    chunks = []
    chunk_duration = 55

    try:
        if duration <= chunk_duration:
            with sr.AudioFile(audio_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.record(source)
            chunks.append(recognizer.recognize_google(audio))
        else:
            offset = 0
            while offset < duration:
                with sr.AudioFile(audio_path) as source:
                    chunk = recognizer.record(
                        source,
                        duration=min(chunk_duration, duration - offset),
                        offset=offset
                    )
                try:
                    chunks.append(recognizer.recognize_google(chunk))
                except sr.UnknownValueError:
                    pass
                offset += chunk_duration

    except sr.UnknownValueError:
        return "", False
    except sr.RequestError as e:
        raise Exception(f"Google Speech API error: {e}")

    result = " ".join(chunks).strip()
    return result, bool(result)


def run_audio_pipeline(video_path, is_video):
    """
    Full audio pipeline: extract → transcribe.
    Returns dict with keys: transcript, speech_found, error
    """
    try:
        if is_video:
            audio_path = extract_audio_from_video(video_path)
        else:
            audio_path = video_path
        transcript, speech_found = transcribe_audio(audio_path)
        if is_video and os.path.exists(audio_path):
            os.remove(audio_path)
        return {"transcript": transcript, "speech_found": speech_found, "error": None}
    except Exception as e:
        return {"transcript": "", "speech_found": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# VISUAL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path, num_frames=8):
    """
    Use ffmpeg to extract evenly-spaced frames from the video.
    """
    frames_dir = video_path + "_frames"
    os.makedirs(frames_dir, exist_ok=True)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    try:
        duration = float(probe.stdout.strip())
    except Exception:
        duration = 30.0

    interval = max(1.0, duration / num_frames)
    subprocess.run(
        ["ffmpeg", "-i", video_path,
         "-vf", f"fps=1/{interval:.1f}",
         "-vframes", str(num_frames),
         os.path.join(frames_dir, "frame_%03d.jpg"), "-y"],
        capture_output=True, text=True
    )

    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    return frames, frames_dir


def analyze_frames_with_blip(frames):
    """
    Run BLIP on each frame to describe what is HAPPENING in the video scene.
    Uses unconditional captioning (no text prompt) so BLIP freely describes
    the scene — people, actions, objects, settings — rather than reading text.
    Returns (combined_description, list_of_captions).
    """
    from PIL import Image
    import torch

    processor, model, device = load_blip()
    dtype = next(model.parameters()).dtype
    captions = []

    for i, frame_path in enumerate(frames):
        try:
            image = Image.open(frame_path).convert("RGB")

            # ── Unconditional captioning: NO text prompt ──────────────────────
            # Passing NO text= forces BLIP to describe the scene freely:
            # people, actions, objects, environment — not just on-screen text.
            inputs = processor(image, return_tensors="pt")
            
            # Move inputs to device and match dtype for pixel_values
            inputs = {k: v.to(device) if k != "pixel_values" else v.to(device, dtype=dtype) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=150,       # allow longer scene descriptions
                    num_beams=7,              # increased beam search for better quality
                    min_length=15,            # avoid trivially short captions
                    repetition_penalty=1.2,   # reduce repetition inside the output
                    no_repeat_ngram_size=2    # further prevent repetitive loops
                )
            caption = processor.decode(output[0], skip_special_tokens=True)

            # Tag each frame with its position in the video for context
            total = len(frames)
            if total > 1:
                pct = int((i / (total - 1)) * 100) if total > 1 else 0
                caption_with_pos = f"[{pct}% into video] {caption}"
            else:
                caption_with_pos = caption

            captions.append(caption_with_pos)

        except Exception as e:
            captions.append(f"[frame {i+1} error: {e}]")

    # Deduplicate: skip captions whose first 60 chars match an existing one
    unique_captions = []
    for c in captions:
        core = c.split("] ", 1)[-1][:60]  # compare content, ignore position tag
        if not any(core in existing for existing in unique_captions):
            unique_captions.append(c)

    # Build a coherent narrative string joining all unique scene descriptions
    combined = ". ".join(unique_captions)
    return combined, unique_captions


def run_visual_pipeline(video_path):
    """
    Full visual pipeline: extract frames → BLIP captions.
    Returns dict with keys: visual_description, frame_captions, frames_dir, error
    """
    frames_dir = None
    try:
        frames, frames_dir = extract_frames(video_path, num_frames=8)
        if not frames:
            return {"visual_description": "", "frame_captions": [],
                    "frames_dir": None, "error": "No frames extracted"}
        visual_description, frame_captions = analyze_frames_with_blip(frames)
        return {
            "visual_description": visual_description,
            "frame_captions": frame_captions,
            "frames_dir": frames_dir,
            "error": None
        }
    except Exception as e:
        return {
            "visual_description": "",
            "frame_captions": [],
            "frames_dir": frames_dir,
            "error": str(e)
        }


# ─────────────────────────────────────────────────────────────────────────────
# FUSION — Merge audio + visual into one rich context
# ─────────────────────────────────────────────────────────────────────────────

def fuse_audio_visual(transcript, speech_found, visual_description):
    if speech_found and visual_description.strip():
        fused = (
            f"SPOKEN AUDIO (what is said in the video):\n{transcript}\n\n"
            f"VISUAL SCENE DESCRIPTIONS (what is happening/shown frame by frame):\n{visual_description}"
        )
        mode = "combined"
    elif speech_found:
        fused = transcript
        mode = "speech"
    elif visual_description.strip():
        fused = visual_description
        mode = "visual"
    else:
        fused = ""
        mode = "none"

    return fused, mode


# ─────────────────────────────────────────────────────────────────────────────
# AI REFERENCE ANSWER GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_reference_answer(fused_content, ai_provider=None, mode="combined"):
    if not ai_provider:
        ai_provider = os.getenv("AI_PROVIDER", "groq")

    if mode == "combined":
        prompt = f"""You are an expert video analyst and evaluator. Below is content extracted from a video — the spoken audio AND visual scene descriptions captured frame by frame.

{fused_content}

Using BOTH what is spoken and what is visually happening in the video, generate a comprehensive REFERENCE DESCRIPTION that:
1. Describes what the video is about and what happens throughout
2. Integrates key spoken information with the visual scenes
3. Explains the topic, activity, or story shown in the video
4. Reads as a complete, well-structured ideal description

Respond ONLY with the reference description, no preamble:"""

    elif mode == "visual":
        prompt = f"""You are an expert video analyst. Below are scene descriptions captured from frames of a silent video (no speech detected).

Visual scene descriptions (frame by frame):
"{fused_content}"

Based on these visual scenes, generate a comprehensive description of:
1. What is happening in this video
2. Who or what appears in the video
3. The overall topic, activity, or story being shown

Respond ONLY with the description, no preamble:"""

    else:  # speech only
        prompt = f"""You are an expert evaluator. Below is a speech transcription from a video.

Generate a comprehensive, ideal REFERENCE ANSWER on this topic:
1. Cover all key concepts mentioned
2. Add important missing information
3. Be clear, accurate, and complete

Transcription:
"{fused_content}"

Respond ONLY with the reference answer, no preamble:"""

    if ai_provider == "groq":
        from openai import OpenAI
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise Exception("Groq API key not configured. Get FREE key at: https://console.groq.com")
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200, temperature=0.3
        )
        return response.choices[0].message.content.strip()

    elif ai_provider == "gemini":
        import requests
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            raise Exception("Gemini API key not configured. Get FREE key at: https://aistudio.google.com/apikey")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}],
                   "generationConfig": {"maxOutputTokens": 1200, "temperature": 0.3}}
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    elif ai_provider == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise Exception("OpenAI API key not configured.")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200, temperature=0.3
        )
        return response.choices[0].message.content.strip()

    elif ai_provider == "anthropic":
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your_anthropic_api_key_here":
            raise Exception("Anthropic API key not configured.")
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()

    else:
        raise Exception(f"Unknown AI provider: '{ai_provider}'. Choose: groq, gemini, openai, or anthropic.")


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────

def compute_similarity(text1, text2):
    """TF-IDF cosine + Jaccard keyword overlap + length ratio → weighted score"""
    if not text1.strip() or not text2.strip():
        return {"overall": 0, "cosine": 0, "keyword": 0, "length_ratio": 0}

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except Exception:
        cosine_sim = 0

    stop_words = {"the","a","an","is","are","was","were","be","been","being","have","has",
                  "had","do","does","did","will","would","could","should","may","might",
                  "shall","can","to","of","in","for","on","with","at","by","from","as",
                  "into","through","and","or"}
    w1 = set(text1.lower().split()) - stop_words
    w2 = set(text2.lower().split()) - stop_words
    jaccard = len(w1 & w2) / len(w1 | w2) if (w1 | w2) else 0

    len_ratio = (min(len(text1), len(text2)) / max(len(text1), len(text2))
                 if max(len(text1), len(text2)) > 0 else 0)

    overall = (cosine_sim * 0.5) + (jaccard * 0.35) + (len_ratio * 0.15)

    return {
        "overall": round(float(overall) * 100, 1),
        "cosine": round(float(cosine_sim) * 100, 1),
        "keyword": round(float(jaccard) * 100, 1),
        "length_ratio": round(float(len_ratio) * 100, 1)
    }


def get_similarity_label(score):
    if score >= 80:
        return ("Excellent", "The explanation closely matches the ideal reference answer.", "#00d4aa")
    elif score >= 60:
        return ("Good", "Most key points covered with minor gaps.", "#f59e0b")
    elif score >= 40:
        return ("Moderate", "Some concepts present but important details missing.", "#f97316")
    elif score >= 20:
        return ("Weak", "Touches on the topic but lacks depth and accuracy.", "#ef4444")
    else:
        return ("Poor", "Content differs significantly from a comprehensive explanation.", "#dc2626")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    ai_provider = request.form.get("ai_provider", os.getenv("AI_PROVIDER", "groq"))
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    frames_dir = None

    try:
        ext = filename.rsplit(".", 1)[1].lower()
        is_video = ext in {"mp4", "avi", "mov", "mkv", "webm"}

        if is_video:
            print("🚀 Running audio and visual pipelines in parallel...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                audio_future  = executor.submit(run_audio_pipeline,  filepath, is_video)
                visual_future = executor.submit(run_visual_pipeline, filepath)

                audio_result  = audio_future.result()
                visual_result = visual_future.result()

            frames_dir = visual_result.get("frames_dir")

            if audio_result["error"] and not visual_result["visual_description"]:
                raise Exception(f"Audio failed: {audio_result['error']} | "
                                f"Visual failed: {visual_result['error']}")

        else:
            audio_result = run_audio_pipeline(filepath, is_video=False)
            visual_result = {"visual_description": "", "frame_captions": [],
                             "frames_dir": None, "error": None}

        fused_content, analysis_mode = fuse_audio_visual(
            transcript         = audio_result["transcript"],
            speech_found       = audio_result["speech_found"],
            visual_description = visual_result["visual_description"]
        )

        if not fused_content.strip():
            raise Exception(
                "Could not extract any content from this file. "
                "Ensure the video has clear speech and/or visible content."
            )

        reference = generate_reference_answer(fused_content, ai_provider, mode=analysis_mode)

        raw_content = audio_result["transcript"] if audio_result["speech_found"] \
                      else visual_result["visual_description"]
        scores = compute_similarity(raw_content, reference)
        label, description, color = get_similarity_label(scores["overall"])

        if os.path.exists(filepath):
            os.remove(filepath)
        if frames_dir and os.path.exists(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)

        return jsonify({
            "success":        True,
            "analysis_mode":  analysis_mode,
            "transcript":     audio_result["transcript"],
            "visual_summary": visual_result["visual_description"],
            "frame_captions": visual_result["frame_captions"],
            "fused_content":  fused_content,
            "reference":      reference,
            "scores":         scores,
            "label":          label,
            "description":    description,
            "color":          color
        })

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        if frames_dir and os.path.exists(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "message": "VidMatch — parallel audio + visual analysis active"
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT — FIX: use_reloader=False prevents WinError 10038 on Windows
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🎬 VidMatch — Video-to-Text Similarity Evaluator")
    print("=" * 52)
    print("⚡ Parallel mode: Audio + Visual run simultaneously")
    print("👁  BLIP analyzes whiteboard/slides while speech is transcribed")
    print("🌐 Running at: http://localhost:5000")
    print("=" * 52 + "\n")

    # FIX for Windows WinError 10038:
    # use_reloader=False prevents Werkzeug's watchdog thread from conflicting
    # with background threads (BLIP model loader) on Windows sockets.
    # debug=True is kept so you still get error tracebacks in the browser.
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        use_reloader=False,   # ← KEY FIX: disables the file watcher thread
        threaded=True         # ← allows concurrent requests
    )