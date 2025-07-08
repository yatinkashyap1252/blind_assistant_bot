#!/bin/bash

echo "🔧 Setting up VS Code for Blind Assistant Project..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv blind_assistant_env

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source blind_assistant_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Install additional development tools
echo "🛠️ Installing development tools..."
pip install black pylint autopep8

echo "✅ VS Code setup complete!"
echo ""
echo "🚀 To run in VS Code:"
echo "1. Open VS Code: code ."
echo "2. Select Python interpreter: Ctrl+Shift+P → 'Python: Select Interpreter'"
echo "3. Choose: ./blind_assistant_env/bin/python"
echo "4. Run: Press F5 or use Run button"
echo ""
echo "📝 Available tasks (Ctrl+Shift+P → 'Tasks: Run Task'):"
echo "   • Run Blind Assistant"
echo "   • Install Dependencies" 
echo "   • Train Models"
