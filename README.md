# 🦮 Blind Assistant - AI Assistive Technology

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![College Project](https://img.shields.io/badge/Project-College%20AI-purple.svg)]()

> An AI-powered assistive technology application designed to help blind and visually impaired individuals navigate their environment using computer vision and machine learning.

## 🎯 Project Overview

This college AI project demonstrates the integration of **5 different AI/ML technologies** into a practical, real-world assistive technology solution. The application provides comprehensive visual analysis through an accessible interface with voice feedback.

## ✨ Key Features

### 🤖 5 AI Modules (100% Working)
- **🔍 Enhanced Object Detection** - Identifies obstacles, furniture, people with safety alerts
- **💰 Advanced Currency Detection** - **88.46% accuracy** for Indian ₹500 & ₹2000 notes
- **🌟 Celebrity Face Recognition** - Recognizes **31 Bollywood & Hollywood celebrities**
- **📖 High-Precision OCR** - Reads text with confidence scores
- **🎨 Comprehensive Color Analysis** - Detailed color information

### 🎭 Celebrity Database
**31 celebrities including:**
- **Bollywood:** Akshay Kumar, Alia Bhatt, Amitabh Bachchan, Hrithik Roshan, Priyanka Chopra
- **Hollywood:** Brad Pitt, Henry Cavill, Tom Cruise, Robert Downey Jr, Margot Robbie
- **Sports:** Virat Kohli, Roger Federer
- **Music:** Billie Eilish, Camila Cabello

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Linux/Ubuntu (recommended)
- Webcam or image files for testing

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/blind-assistant-ai.git
cd blind-assistant-ai
```

2. **Create virtual environment**
```bash
python3 -m venv blind_assistant_env
source blind_assistant_env/bin/activate  # Linux/Mac
# or
blind_assistant_env\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required models** (see [Model Setup](#-model-setup))

5. **Run the application**
```bash
cd src
python3 blind_assistant_final.py
```

## 📦 Model Setup

Due to GitHub file size limitations, download these models separately:

### Required Models
1. **YOLO Object Detection Model**
   ```bash
   # Download YOLOv8s model (22MB)
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
   # Place in: src/yolov8s.pt
   ```

2. **Celebrity Face Encodings**
   - The application will automatically create celebrity encodings on first run
   - Or download pre-trained encodings: [Celebrity Models](https://drive.google.com/your-link)

### Optional Datasets
- **Celebrity Faces:** [Celebrity Dataset](https://www.kaggle.com/datasets/your-dataset-link)
- **Currency Images:** [Indian Currency Dataset](https://github.com/your-dataset-link)

> **Note:** The application works without datasets - it will create sample data for demonstration.

## 🏗️ Project Structure

```
blind-assistant-ai/
├── src/                                    # Source code
│   ├── blind_assistant_final.py          # 🎯 Main application
│   ├── improved_object_detection.py      # Enhanced object detection
│   ├── enhanced_currency_detection.py    # 88.46% accuracy currency detection
│   ├── celebrity_face_recognition.py     # Celebrity face recognition
│   ├── ocr_module.py                     # OCR text detection
│   └── color_detection.py               # Color analysis
├── models/                               # Trained models (download separately)
├── docs/                                # Documentation
├── requirements.txt                     # Python dependencies
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

## 🛠️ Technology Stack

- **Computer Vision:** OpenCV, YOLO
- **Machine Learning:** Scikit-learn, Face Recognition
- **GUI:** Tkinter
- **Text Processing:** Tesseract OCR
- **Audio:** pyttsx3 (Text-to-Speech)
- **Image Processing:** PIL, NumPy

## 📊 Performance Metrics

- **Currency Detection:** 88.46% accuracy (improved from 57%)
- **Object Detection:** Enhanced with safety alerts
- **Face Recognition:** 31 celebrities with similarity matching
- **OCR:** 95% confidence text recognition
- **Overall Success Rate:** 5/5 modules working (100%)

## 🎮 Usage

1. **Launch Application**
   ```bash
   python3 src/blind_assistant_final.py
   ```

2. **Load Image**
   - Click "Load Image" to select your image
   - Or click "Load Sample" for demonstration

3. **Choose AI Analysis**
   - Object Detection - Identify objects and obstacles
   - Currency Detection - Recognize Indian currency notes
   - Celebrity Recognition - Match faces with celebrities
   - OCR Text Reading - Extract text from images
   - Color Analysis - Analyze colors and properties

4. **Get Results**
   - View detailed analysis in the results panel
   - Use "Speak Results" for audio feedback

## 🎓 Academic Highlights

### Technical Achievements
- **Multi-modal AI Integration** - 5 different AI technologies
- **Custom Model Training** - Enhanced currency detection with data augmentation
- **Real-world Application** - Practical assistive technology
- **Accessibility Focus** - Voice-first interface design
- **Performance Optimization** - Works on single-core systems

### Innovation Points
- **88.46% currency accuracy** - Significant improvement through feature engineering
- **Celebrity recognition** - Custom face encoding system
- **Safety alerts** - Object detection with navigation assistance
- **Professional GUI** - User-friendly interface for demonstrations

## 🤝 Contributing

This is a college project, but suggestions and improvements are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV Community** - Computer vision libraries
- **Ultralytics** - YOLO object detection models
- **Face Recognition Library** - Face encoding algorithms
- **Tesseract OCR** - Text recognition engine
- **College Faculty** - Guidance and support

## 📞 Contact

- **Project Author:** [Your Name]
- **Email:** [your.email@college.edu]
- **College:** [Your College Name]
- **Course:** AI/ML Project

## 🔗 Links

- **Demo Video:** [YouTube Link]
- **Project Presentation:** [Slides Link]
- **Dataset Sources:** [Kaggle/GitHub Links]
- **Model Downloads:** [Google Drive/GitHub Releases]

---

**🎯 College AI Project - Demonstrating practical applications of AI/ML for assistive technology**

⭐ **Star this repository if you found it helpful!**
