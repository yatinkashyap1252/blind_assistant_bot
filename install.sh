#!/bin/bash

echo "🚀 Setting up Blind Assistant Project..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt update

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y python3-pip python3-dev python3-tk
sudo apt install -y tesseract-ocr tesseract-ocr-eng
sudo apt install -y libgl1-mesa-glx libglib2.0-0
sudo apt install -y espeak espeak-data libespeak1 libespeak-dev
sudo apt install -y portaudio19-dev python3-pyaudio

# Install Python packages
echo "🐍 Installing Python packages..."
pip3 install --upgrade pip

# Install packages one by one to handle potential conflicts
echo "Installing OpenCV..."
pip3 install opencv-python==4.8.1.78

echo "Installing NumPy and Pandas..."
pip3 install numpy==1.24.3 pandas==2.0.3

echo "Installing Scikit-learn..."
pip3 install scikit-learn==1.3.0

echo "Installing TensorFlow..."
pip3 install tensorflow==2.13.0

echo "Installing OCR dependencies..."
pip3 install pytesseract==0.3.10

echo "Installing Text-to-Speech..."
pip3 install pyttsx3==2.90

echo "Installing Face Recognition..."
pip3 install face-recognition==1.3.0 dlib==19.24.2

echo "Installing YOLO..."
pip3 install ultralytics==8.0.196

echo "Installing MediaPipe..."
pip3 install mediapipe==0.10.3

echo "Installing Image Processing..."
pip3 install Pillow==10.0.0

echo "Installing Plotting libraries..."
pip3 install matplotlib==3.7.2 seaborn==0.12.2

echo "Installing additional dependencies..."
pip3 install webcolors playsound==1.3.0

# Create desktop shortcut
echo "🖥️ Creating desktop shortcut..."
cat > ~/Desktop/BlindAssistant.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Blind Assistant
Comment=AI Vision Helper for Blind People
Exec=python3 /home/yatin/blind_assistant_project/src/main.py
Icon=applications-accessibility
Terminal=false
Categories=Accessibility;
EOF

chmod +x ~/Desktop/BlindAssistant.desktop

echo "✅ Installation complete!"
echo ""
echo "🎯 To run the application:"
echo "   cd /home/yatin/blind_assistant_project/src"
echo "   python3 main.py"
echo ""
echo "📝 Notes:"
echo "   • Add photos to datasets/faces/known_faces/ for face recognition"
echo "   • The app will train models on first run (may take a few minutes)"
echo "   • Use good lighting for better detection accuracy"
echo ""
echo "🚀 Ready to help blind users navigate the world!"
