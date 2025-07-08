#!/bin/bash
# 🦮 Blind Assistant - Automated GitHub Setup Script

echo "🚀 BLIND ASSISTANT - GITHUB SETUP"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git is not installed. Please install git first.${NC}"
    echo "   Ubuntu/Debian: sudo apt-get install git"
    exit 1
fi

echo -e "${GREEN}✅ Git is installed${NC}"

# Get user information
echo -e "\n${BLUE}📝 Please provide your information:${NC}"
read -p "Your Name: " USER_NAME
read -p "Your Email: " USER_EMAIL
read -p "Your GitHub Username: " GITHUB_USERNAME
read -p "Repository Name (default: blind-assistant-ai): " REPO_NAME

# Set default repo name if empty
if [ -z "$REPO_NAME" ]; then
    REPO_NAME="blind-assistant-ai"
fi

# Configure git
echo -e "\n${YELLOW}🔧 Configuring Git...${NC}"
git config --global user.name "$USER_NAME"
git config --global user.email "$USER_EMAIL"

# Initialize repository
echo -e "\n${YELLOW}📂 Initializing Git repository...${NC}"
git init

# Add all files
echo -e "\n${YELLOW}📁 Adding files to repository...${NC}"
git add .

# Show what will be uploaded
echo -e "\n${BLUE}📋 Files to be uploaded:${NC}"
git status --porcelain | head -20

# Calculate upload size
UPLOAD_SIZE=$(git ls-files | xargs du -ch 2>/dev/null | tail -1 | cut -f1)
echo -e "\n${GREEN}📊 Total upload size: ${UPLOAD_SIZE}${NC}"

# Create first commit
echo -e "\n${YELLOW}💾 Creating first commit...${NC}"
git commit -m "🦮 Initial commit: Blind Assistant AI Project

✨ Features:
- 5 AI modules (Object Detection, Currency, Face Recognition, OCR, Color)
- 88.46% currency detection accuracy  
- 31 celebrity recognition database
- Professional GUI with voice feedback
- College AI project submission

🎯 Ready for demonstration and evaluation

👨‍🎓 Author: $USER_NAME
📧 Contact: $USER_EMAIL"

# Add remote repository
echo -e "\n${YELLOW}🔗 Adding GitHub remote...${NC}"
git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git

echo -e "\n${GREEN}✅ Git repository setup complete!${NC}"

echo -e "\n${BLUE}📋 NEXT STEPS:${NC}"
echo "1. Go to https://github.com and create a new repository named: $REPO_NAME"
echo "2. Make sure it's PUBLIC for college submission"
echo "3. Don't initialize with README (we have our own files)"
echo "4. After creating the repository, run:"
echo -e "   ${YELLOW}git push -u origin main${NC}"

echo -e "\n${BLUE}🎯 Your GitHub repository will be:${NC}"
echo "   https://github.com/$GITHUB_USERNAME/$REPO_NAME"

echo -e "\n${GREEN}🎉 Ready for college submission!${NC}"

# Create model download script
echo -e "\n${YELLOW}📥 Creating model download script...${NC}"
cat > download_models.sh << 'EOF'
#!/bin/bash
# 🦮 Blind Assistant - Model Download Script

echo "🚀 Downloading required models for Blind Assistant..."

# Create directories
mkdir -p src
mkdir -p models/custom

# Download YOLO model (22MB)
echo "📥 Downloading YOLO object detection model..."
cd src

if command -v wget &> /dev/null; then
    wget -O yolov8s.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
elif command -v curl &> /dev/null; then
    curl -L -o yolov8s.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
else
    echo "❌ Neither wget nor curl found. Please download manually:"
    echo "   URL: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
    echo "   Save as: src/yolov8s.pt"
    exit 1
fi

if [ $? -eq 0 ]; then
    echo "✅ YOLO model downloaded successfully!"
    echo "📊 Model size: $(du -sh yolov8s.pt | cut -f1)"
else
    echo "❌ Failed to download YOLO model. Please download manually."
    echo "   URL: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
    echo "   Save as: src/yolov8s.pt"
fi

cd ..
echo "🎉 Setup complete! You can now run the application."
echo "   Command: python3 src/blind_assistant_final.py"
EOF

chmod +x download_models.sh

# Add the download script to git
git add download_models.sh
git commit -m "📥 Add automated model download script"

echo -e "\n${GREEN}✅ Model download script created!${NC}"
echo -e "\n${BLUE}📋 SUMMARY:${NC}"
echo "- Repository initialized with all source code"
echo "- Professional documentation included"
echo "- Large files excluded (models, datasets)"
echo "- Model download script created"
echo "- Ready for GitHub upload"

echo -e "\n${YELLOW}🎓 For College Submission:${NC}"
echo "1. Push to GitHub: git push -u origin main"
echo "2. Share repository URL with professors"
echo "3. Repository includes complete setup instructions"
echo "4. All 5 AI modules ready for demonstration"

echo -e "\n${GREEN}🏆 Your Blind Assistant project is ready for GitHub!${NC}"
