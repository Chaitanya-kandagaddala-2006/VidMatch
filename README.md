# VidMatch

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=VidMatch&fontSize=80&fontColor=ffffff&fontAlignY=38&desc=Video-to-Text%20Similarity%20Evaluator&descAlignY=60&descSize=20&descColor=a78bfa" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![ffmpeg](https://img.shields.io/badge/ffmpeg-required-007808?style=for-the-badge&logo=ffmpeg&logoColor=white)](https://ffmpeg.org)
[![BLIP](https://img.shields.io/badge/BLIP-Vision_AI-a78bfa?style=for-the-badge)](https://huggingface.co/Salesforce/blip-image-captioning-large)

<br/>

**Upload a video → Extract speech & visuals → Score against an AI reference answer**

*Powered by Google Speech, BLIP Vision AI, and your choice of 4 free LLM providers*

<br/>

[🚀 Quick Start](#-quick-start) · [⚙️ How It Works](#️-how-it-works) · [📊 Scoring](#-similarity-scoring) · [🛠 Troubleshooting](#-troubleshooting)

<br/>

[![GitHub followers](https://img.shields.io/github/followers/Chaitanya-kandagaddala-2006?label=Follow&style=social)](https://github.com/Chaitanya-kandagaddala-2006)
[![GitHub stars](https://img.shields.io/github/stars/Chaitanya-kandagaddala-2006/vidmatch?style=social)](https://github.com/Chaitanya-kandagaddala-2006/vidmatch)

</div>

---

## 🎯 What is VidMatch?

VidMatch is a Flask web app that evaluates how well a video explains a topic. It:

1. **Transcribes** speech from your video using Google's free Speech Recognition API
2. **Describes** visual content (whiteboards, slides, demos) using the BLIP Large vision model — locally, no API needed
3. **Fuses** both outputs and generates an *ideal* reference answer using an AI provider of your choice
4. **Scores** your video against that reference using TF-IDF cosine similarity, keyword overlap, and length ratio

Perfect for educators, students, and content creators who want objective feedback on their explanations.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 🎙 **Speech Transcription** | Google Speech Recognition — completely free, auto-chunked for long videos |
| 👁 **Visual Analysis** | BLIP Large model runs locally — no API key, no cost, works offline |
| ⚡ **Parallel Processing** | Audio and visual pipelines run simultaneously for maximum speed |
| 🤖 **4 AI Providers** | Groq, Gemini, OpenAI, Anthropic — all with free tiers |
| 📁 **Multi-Format** | MP4, AVI, MOV, MKV, WEBM, MP3, WAV, M4A, FLAC |
| 📊 **Detailed Scoring** | 3 metrics with animated score ring and color-coded grade |
| 🔒 **Privacy First** | Uploaded files are deleted immediately after processing |
| 🖥 **Single-File Frontend** | Pure HTML/CSS/JS — no Node.js, no build step |

---

## 🚀 Quick Start

### Prerequisites

Make sure these are installed before you begin:

```bash
# Check Python (3.8+ required)
python --version

# Check ffmpeg
ffmpeg -version
```

> **Windows:** Install ffmpeg with `winget install ffmpeg` and restart your terminal.  
> **Mac:** `brew install ffmpeg` · **Linux:** `sudo apt install ffmpeg`

---

### 1. Clone the repo

```bash
git clone https://github.com/Chaitanya-kandagaddala-2006/vidmatch.git
cd vidmatch
```

### 2. Run setup

```bash
python setup.py
```

This installs all dependencies and creates your `.env` config file from the template.

### 3. Get a free API key

Choose **one** provider (Groq recommended — fastest and free):

| Provider | Free? | Sign Up |
|----------|-------|---------|
| ⭐ **Groq** | ✅ Free | [console.groq.com](https://console.groq.com) |
| **Gemini** | ✅ Free | [aistudio.google.com](https://aistudio.google.com/apikey) |
| **OpenAI** | 💳 ~$5 credits | [platform.openai.com](https://platform.openai.com) |
| **Anthropic** | 💳 ~$5 credits | [console.anthropic.com](https://console.anthropic.com) |

### 4. Configure `.env`

```env
AI_PROVIDER=groq

GROQ_API_KEY=gsk_your_key_here
# GEMINI_API_KEY=AIza_your_key_here
# OPENAI_API_KEY=sk-your_key_here
# ANTHROPIC_API_KEY=sk-ant-your_key_here
```

### 5. Launch

```bash
python run.py
```

Open **http://localhost:5000** in your browser. 🎉

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────┐
│                   Video / Audio File                │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
  🎙 AUDIO PIPELINE          👁 VISUAL PIPELINE
  ─────────────────          ──────────────────
  ffmpeg → WAV               ffmpeg → frames
  Google Speech API          BLIP Large (local)
  → Transcript               → Scene description
          │                         │
          └────────────┬────────────┘
                       ▼
              🔀 CONTENT FUSION
              ─────────────────
              Audio + Visual merged
              (or solo if one fails)
                       │
                       ▼
          🤖 AI REFERENCE GENERATION
          ────────────────────────────
          Groq / Gemini / OpenAI / Anthropic
          → Ideal explanation of the content
                       │
                       ▼
            📊 SIMILARITY SCORING
            ──────────────────────
            TF-IDF Cosine  (50%)
            Jaccard Keywords (35%)
            Length Ratio   (15%)
                       │
                       ▼
            🖥 RESULTS DASHBOARD
```

---

## 📊 Similarity Scoring

Three metrics are computed and combined into a single weighted score:

| Metric | Method | Weight |
|--------|--------|--------|
| **Semantic** | TF-IDF Cosine Similarity | 50% |
| **Keyword** | Jaccard Overlap (stopwords removed) | 35% |
| **Coverage** | Length Ratio | 15% |

### Grade Scale

```
  ████████████████████  80–100%  🟢  Excellent  — Closely matches the ideal answer
  ████████████████      60–79%   🟡  Good       — Covers most key points
  ████████████          40–59%   🟠  Moderate   — Some concepts present, gaps exist
  ████████              20–39%   🔴  Weak       — Lacks depth and accuracy
  ████                  0–19%    ⛔  Poor       — Significantly different
```

---

## 🗂 Project Structure

```
vidmatch/
├── 📄 app.py              # Flask backend — audio, visual, scoring, routes
├── 🚀 run.py              # Server launcher with provider validation
├── ⚙️  setup.py            # One-time dependency installer
├── 📦 requirements.txt    # Python packages
├── 🔐 .env.example        # Config template (copy → .env)
├── 🔐 .env                # Your secrets (git-ignored)
└── templates/
    └── 🌐 index.html      # Full frontend — drag-drop UI, score ring, results
```

---

## 📦 Requirements

```
flask >= 2.3.0
openai >= 1.0.0
moviepy >= 1.0.3
SpeechRecognition >= 3.10.0
pydub >= 0.25.1
scikit-learn >= 1.3.0
numpy >= 1.24.0
requests >= 2.31.0
python-dotenv >= 1.0.0
werkzeug >= 2.3.0
transformers >= 4.30.0     # BLIP visual model
torch >= 2.0.0             # BLIP inference
Pillow >= 10.0.0           # Frame processing
```

> All installed automatically via `python setup.py`

---

## 🆓 API Cost Reference

| Service | Cost | Notes |
|---------|------|-------|
| Google Speech Recognition | **Free** | 60s per request, auto-chunked |
| Groq (LLaMA 3) | **Free** | Generous rate limits |
| Gemini Flash | **Free** | 15 RPM free tier |
| OpenAI GPT-3.5 Turbo | ~$5 free credits | Pay-as-you-go after |
| Anthropic Claude Haiku | ~$5 free credits | Pay-as-you-go after |
| BLIP Large (Vision AI) | **Free** | Runs locally, ~1.9GB one-time download |

---

## 🛠 Troubleshooting

<details>
<summary><b>❌ ffmpeg not found</b></summary>

Install ffmpeg and ensure it's in your system PATH:
- **Windows:** `winget install ffmpeg` → restart terminal
- **Mac:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

</details>

<details>
<summary><b>❌ Could not understand audio</b></summary>

- Ensure the video has clear English speech
- Heavy background noise reduces transcription accuracy
- Try trimming silent segments before uploading

</details>

<details>
<summary><b>❌ API key not configured</b></summary>

Open `.env` and verify:
- The correct `AI_PROVIDER` is set (e.g., `groq`)
- The matching key variable is filled in
- No extra spaces or quotes around the key value

</details>

<details>
<summary><b>❌ Port already in use</b></summary>

Edit `run.py` and change:
```python
app.run(port=5000)  →  app.run(port=5001)
```

</details>

<details>
<summary><b>⏳ Large files are slow</b></summary>

- Audio: Google processes in 55-second chunks — a 10-min video takes ~2 minutes
- BLIP: First run downloads the model (~1.9GB), subsequent runs are instant
- GPU available? BLIP automatically uses CUDA for much faster frame analysis

</details>

<details>
<summary><b>🪟 WinError 10038 on Windows</b></summary>

Already handled in `app.py` — `use_reloader=False` prevents Werkzeug's watchdog thread from conflicting with BLIP's background loader on Windows sockets.

</details>

---



## 👨‍💻 Author

**Chaitanya Kandagaddala**  
[![GitHub](https://img.shields.io/badge/GitHub-Chaitanya--kandagaddala--2006-181717?style=flat&logo=github)](https://github.com/Chaitanya-kandagaddala-2006)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with 🎬 Python · Flask · BLIP · Google Speech · scikit-learn

<br/>

⭐ **Star this repo if you found it useful!**

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=100&section=footer" width="100%"/>

</div>
