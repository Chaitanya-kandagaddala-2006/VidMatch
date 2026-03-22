#!/usr/bin/env python3
"""
VidMatch Setup Script
Run this once to set up the project, then use run.py to start the server.
"""

import subprocess
import sys
import os
import shutil

def check_python():
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8+ is required!")
        sys.exit(1)

def check_ffmpeg():
    if shutil.which("ffmpeg"):
        print("✓ ffmpeg found")
        return True
    else:
        print("✗ ffmpeg NOT found!")
        print("\n📦 Install ffmpeg:")
        print("  Windows: https://ffmpeg.org/download.html or 'winget install ffmpeg'")
        print("  Mac:     brew install ffmpeg")
        print("  Linux:   sudo apt install ffmpeg")
        print("\nffmpeg is required for video audio extraction.")
        return False

def install_requirements():
    print("\n📦 Installing Python packages...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=False
    )
    if result.returncode == 0:
        print("✓ All packages installed!")
    else:
        print("✗ Some packages failed to install. Check errors above.")
        sys.exit(1)

def setup_env():
    if not os.path.exists(".env"):
        shutil.copy(".env.example", ".env")
        print("\n✓ Created .env file from template")
        print("⚠️  IMPORTANT: Edit .env and add your API key!")
        print("   Get a FREE OpenAI key: https://platform.openai.com")
        print("   Get a FREE Anthropic key: https://console.anthropic.com")
    else:
        print("✓ .env file already exists")

def main():
    print("=" * 50)
    print("  VidMatch — Setup Wizard")
    print("=" * 50 + "\n")

    check_python()
    ffmpeg_ok = check_ffmpeg()
    install_requirements()
    setup_env()

    print("\n" + "=" * 50)
    if ffmpeg_ok:
        print("✅ Setup complete! Run: python run.py")
    else:
        print("⚠️  Setup done, but install ffmpeg first!")
        print("   Then run: python run.py")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
