#!/usr/bin/env python3
"""
VidMatch — Run Server
Simply run: python run.py
"""
import os
import sys

# Check .env exists
if not os.path.exists(".env"):
    print("⚠️  .env file not found. Run 'python setup.py' first!")
    sys.exit(1)

# Check API key is configured
from dotenv import load_dotenv
load_dotenv()

provider = os.getenv("AI_PROVIDER", "groq")

KEY_CONFIGS = {
    "groq":      ("GROQ_API_KEY",      "your_groq_api_key_here",      "https://console.groq.com",           "gsk_..."),
    "gemini":    ("GEMINI_API_KEY",    "your_gemini_api_key_here",    "https://aistudio.google.com/apikey", "AIza..."),
    "openai":    ("OPENAI_API_KEY",    "your_openai_api_key_here",    "https://platform.openai.com",        "sk-..."),
    "anthropic": ("ANTHROPIC_API_KEY", "your_anthropic_api_key_here", "https://console.anthropic.com",      "sk-ant-..."),
}

if provider not in KEY_CONFIGS:
    print(f"⚠️  Unknown AI_PROVIDER '{provider}' in .env. Choose: groq, gemini, openai, anthropic")
    sys.exit(1)

env_var, placeholder, signup_url, key_format = KEY_CONFIGS[provider]
key = os.getenv(env_var, "")
if not key or key == placeholder:
    free_label = "(FREE - no credit card)" if provider in ("groq", "gemini") else "(paid / signup credits)"
    print(f"⚠️  {env_var} not set in .env file!")
    print(f"   Provider: {provider.upper()} {free_label}")
    print(f"   Sign up:  {signup_url}")
    print(f"   Key format: {key_format}")
    print(f"   Then edit .env: {env_var}=your-key-here")
    sys.exit(1)

print("\n🎬 VidMatch — Video-to-Text Similarity Evaluator")
print("=" * 50)
print(f"🤖 AI Provider: {provider.upper()}")
print("🌐 Server: http://localhost:5000")
print("📁 Open the URL in your browser (Chrome/Edge recommended)")
print("⛔ Press Ctrl+C to stop")
print("=" * 50 + "\n")

from app import app
app.run(debug=False, host="127.0.0.1", port=5000)
