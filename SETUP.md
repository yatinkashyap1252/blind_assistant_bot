# 🚀 Blind Assistant - Setup Instructions

## 📋 Quick Setup Guide

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/blind-assistant-ai.git
cd blind-assistant-ai
```

### 2. System Dependencies (Linux/Ubuntu)
```bash
# Install system packages
sudo apt-get update
sudo apt-get install python3-pip python3-venv python3-tk tesseract-ocr cmake

# For face recognition (if needed)
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev
```

### 3. Python Environment
```bash
# Create virtual environment
python3 -m venv blind_assistant_env

# Activate environment
source blind_assistant_env/bin/activate  # Linux/Mac
# or
blind_assistant_env\Scripts\activate     # Windows

# Install Python packages
pip install -r requirements.txt
```

### 4. Download Models
```bash
# Create models directory
mkdir -p models/custom

# Download YOLO model (22MB)
cd src
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

# Return to project root
cd ..
```

### 5. Run Application
```bash
# Activate environment (if not already active)
source blind_assistant_env/bin/activate

# Run the application
cd src
python3 blind_assistant_final.py
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Face Recognition Installation Error
```bash
# If face_recognition fails to install
pip install cmake
pip install dlib
pip install face_recognition
```

#### 2. Tesseract OCR Not Found
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# CentOS/RHEL
sudo yum install tesseract

# macOS
brew install tesseract
```

#### 3. Tkinter GUI Error
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# CentOS/RHEL
sudo yum install tkinter
```

#### 4. OpenCV Import Error
```bash
# Reinstall OpenCV
pip uninstall opencv-python
pip install opencv-python==4.8.1.78
```

### 5. YOLO Model Download Issues
If automatic download fails:
1. Manually download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
2. Place in `src/yolov8s.pt`

## 🎯 First Run

### What Happens on First Launch:
1. **Celebrity Face Encodings** - Will be created automatically (may take 2-3 minutes)
2. **Currency Model Training** - Will train on available data
3. **Sample Image** - Will be created for demonstration

### Expected Output:
```
🚀 Initializing AI Modules...
✅ Enhanced Object Detection loaded
✅ Enhanced Currency Detection loaded (88.46% accuracy)
✅ OCR Text Reading loaded
✅ Color Analysis loaded
✅ Celebrity Face Recognition loaded (31 celebrities)
✅ Text-to-Speech initialized
```

## 📊 Performance Notes

- **First run** may take 2-3 minutes to initialize
- **Celebrity recognition** requires initial encoding creation
- **Currency detection** trains automatically on first use
- **Subsequent runs** are much faster (models cached)

## 🎮 Usage Tips

1. **Start with Sample Image** - Click "Load Sample" for quick demo
2. **Try Different Modules** - Test all 5 AI features
3. **Use Voice Feedback** - Click "Speak Results" for audio
4. **Load Your Images** - Test with your own photos

## 🆘 Need Help?

If you encounter issues:
1. Check the error message in terminal
2. Ensure all system dependencies are installed
3. Verify Python version (3.8+ required)
4. Check virtual environment is activated
5. Try reinstalling problematic packages

## 🎓 For College Demonstration

### Quick Demo Setup:
```bash
# One-time setup
git clone [your-repo-url]
cd blind-assistant-ai
python3 -m venv blind_assistant_env
source blind_assistant_env/bin/activate
pip install -r requirements.txt

# For each demo
source blind_assistant_env/bin/activate
cd src
python3 blind_assistant_final.py
```

### Demo Flow:
1. Show welcome screen with features
2. Load sample image
3. Demonstrate each AI module
4. Show results with voice feedback
5. Highlight 88.46% currency accuracy
6. Show celebrity recognition with similarity scores

---

**🎯 Ready to impress your professors!** 🏆
