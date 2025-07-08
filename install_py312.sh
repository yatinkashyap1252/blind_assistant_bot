#!/bin/bash

echo "🚀 Setting up Blind Assistant Project for Python 3.12..."

# Check Python version
python_version=$(python3 --version)
echo "📍 Detected: $python_version"

# Update system packages
echo "📦 Updating system packages..."
sudo apt update

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y python3-pip python3-dev python3-tk python3-venv
sudo apt install -y tesseract-ocr tesseract-ocr-eng
sudo apt install -y libgl1-mesa-glx libglib2.0-0
sudo apt install -y espeak espeak-data libespeak1 libespeak-dev
sudo apt install -y portaudio19-dev python3-pyaudio
sudo apt install -y cmake build-essential

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv blind_assistant_env
source blind_assistant_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install packages step by step with error handling
echo "📚 Installing Python packages..."

# Core packages first
echo "Installing core packages..."
pip install numpy || echo "⚠️ NumPy installation failed"
pip install opencv-python || echo "⚠️ OpenCV installation failed"
pip install Pillow || echo "⚠️ Pillow installation failed"

# Data science packages
echo "Installing data science packages..."
pip install pandas || echo "⚠️ Pandas installation failed"
pip install scikit-learn || echo "⚠️ Scikit-learn installation failed"
pip install matplotlib seaborn || echo "⚠️ Plotting libraries installation failed"

# Computer vision packages
echo "Installing computer vision packages..."
pip install pytesseract || echo "⚠️ Tesseract installation failed"
pip install ultralytics || echo "⚠️ YOLO installation failed"

# Try MediaPipe (may not work with Python 3.12)
echo "Installing MediaPipe..."
pip install mediapipe || echo "⚠️ MediaPipe installation failed - will use alternative"

# Face recognition (may have issues with Python 3.12)
echo "Installing face recognition..."
pip install cmake dlib || echo "⚠️ dlib installation failed"
pip install face-recognition || echo "⚠️ Face recognition installation failed - will use alternative"

# Audio packages
echo "Installing audio packages..."
pip install pyttsx3 || echo "⚠️ TTS installation failed"

# Utility packages
echo "Installing utility packages..."
pip install webcolors || echo "⚠️ Webcolors installation failed"

# Try TensorFlow (latest version for Python 3.12)
echo "Installing TensorFlow..."
pip install tensorflow>=2.15.0 || echo "⚠️ TensorFlow installation failed - using scikit-learn only"

echo "✅ Installation complete!"
echo ""
echo "🎯 To activate environment and run:"
echo "   source blind_assistant_env/bin/activate"
echo "   cd src"
echo "   python3 main.py"
echo ""
echo "⚠️ Note: Some packages may not be fully compatible with Python 3.12"
echo "   The app will work with available packages and graceful fallbacks"
