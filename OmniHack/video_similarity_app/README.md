# 🎬 VidMatch — Video-to-Text Similarity Evaluator

Upload a video, extract the speech automatically, compare it to an AI-generated ideal reference answer, and get a similarity score. All free APIs.

---

## 🚀 Quick Start (VS Code)

### Step 1 — Prerequisites

Install these before starting:

| Tool | Download |
|------|----------|
| Python 3.8+ | https://www.python.org/downloads/ |
| ffmpeg | https://ffmpeg.org/download.html (or `winget install ffmpeg`) |

**Verify in terminal:**
```bash
python --version   # Should show 3.8+
ffmpeg -version    # Should show version info
```

---

### Step 2 — Get a FREE API Key

Choose **one** of these (both have free tiers):

**Option A — OpenAI (GPT-3.5)**
1. Go to https://platform.openai.com
2. Sign up → API Keys → Create new key
3. Copy the key (starts with `sk-...`)

**Option B — Anthropic Claude Haiku**
1. Go to https://console.anthropic.com
2. Sign up → API Keys → Create key
3. Copy the key (starts with `sk-ant-...`)

---

### Step 3 — Setup

Open VS Code terminal (`Ctrl + `` ` ``), navigate to this folder:

```bash
cd path/to/video_similarity_app
python setup.py
```

This installs all packages and creates your `.env` file.

---

### Step 4 — Configure API Key

Open the `.env` file and fill in your key:

```env
# If using OpenAI:
OPENAI_API_KEY=sk-your-key-here
AI_PROVIDER=openai

# If using Anthropic:
ANTHROPIC_API_KEY=sk-ant-your-key-here
AI_PROVIDER=anthropic
```

---

### Step 5 — Run

```bash
python run.py
```

Open your browser to: **http://localhost:5000**

---

## 🎯 How It Works

```
Video/Audio File
      ↓
  Extract Audio (ffmpeg)
      ↓
  Transcribe Speech (Google Speech Recognition — FREE)
      ↓
  Generate Reference Answer (OpenAI GPT / Claude — FREE tier)
      ↓
  Compute Similarity Score (TF-IDF + Keyword Overlap)
      ↓
  Display Results with Score Ring + Reference Dropdown
```

---

## 📊 Similarity Metrics

| Metric | Method | Weight |
|--------|--------|--------|
| Semantic | TF-IDF Cosine Similarity | 50% |
| Keyword | Jaccard Overlap | 35% |
| Coverage | Length Ratio | 15% |

### Score Interpretation

| Score | Label | Meaning |
|-------|-------|---------|
| 80–100% | Excellent | Very close to ideal explanation |
| 60–79% | Good | Covers most key points |
| 40–59% | Moderate | Some concepts present, gaps exist |
| 20–39% | Weak | Lacks depth |
| 0–19% | Poor | Significantly different |

---

## 💡 Features

- **Drag & drop** file upload (video or audio)
- **Multi-format support**: MP4, AVI, MOV, MKV, WEBM, MP3, WAV, M4A, FLAC
- **Long video support**: Automatically chunks audio for Google's free API
- **AI Reference Dropdown**: Click to expand the full AI-generated ideal answer
- **Floating bubble**: Quick access to reference answer from anywhere on the page
- **3 similarity metrics** with animated score ring
- **Side-by-side** transcript vs reference preview

---

## 🛠 Project Structure

```
video_similarity_app/
├── app.py              # Flask backend (main logic)
├── run.py              # Server launcher
├── setup.py            # One-time setup script
├── requirements.txt    # Python dependencies
├── .env.example        # Config template
├── .env                # Your config (created by setup.py)
├── templates/
│   └── index.html      # Full frontend (HTML/CSS/JS)
└── uploads/            # Temp folder (auto-cleaned)
```

---

## ⚠️ Troubleshooting

**"ffmpeg not found"**
→ Install ffmpeg and ensure it's in your system PATH. Restart VS Code after installing.

**"Could not understand audio"**
→ Ensure the video has clear English speech. Background noise may affect accuracy.

**"OpenAI API key not configured"**
→ Check your `.env` file. Make sure there are no extra spaces around the key.

**Port already in use**
→ Change port in `run.py`: `app.run(port=5001)`

**Large files are slow**
→ Google's free Speech API processes in 55-second chunks. A 10-minute video takes ~2 minutes.

---

## 🆓 Free API Usage

| API | Free Tier | Limit |
|-----|-----------|-------|
| Google Speech Recognition | ✅ Free | 60s per request (chunked automatically) |
| OpenAI GPT-3.5 Turbo | ✅ Free credits on signup | ~$5 free credits |
| Anthropic Claude Haiku | ✅ Free credits on signup | ~$5 free credits |

---

## 📝 Requirements

```
flask
openai
anthropic
moviepy
SpeechRecognition
pydub
scikit-learn
numpy
python-dotenv
werkzeug
```
