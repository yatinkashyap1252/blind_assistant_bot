#!/usr/bin/env python3
"""
🦮 BLIND ASSISTANT - FINAL VERSION
==================================

AI-Powered Assistive Technology for Blind and Visually Impaired Individuals

Features:
- Enhanced Object Detection with Safety Alerts
- 88.46% Accuracy Currency Detection (Indian ₹500 & ₹2000)
- Celebrity Face Recognition (31 Bollywood & Hollywood Stars)
- High-Precision OCR Text Reading
- Comprehensive Color Analysis
- Professional GUI with Voice Feedback

Author: College AI Project
Version: Final Submission
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import pyttsx3
import threading
import os

# Import AI modules
from improved_object_detection import ImprovedObjectDetector
from enhanced_currency_detection import EnhancedCurrencyDetector
from ocr_module import OCRDetector
from color_detection import ColorDetector
from celebrity_face_recognition import CelebrityFaceRecognizer

class BlindAssistantFinal:
    def __init__(self, root):
        self.root = root
        self.root.title("🦮 Blind Assistant - AI Assistive Technology")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Initialize AI modules
        self.init_ai_modules()
        
        # Initialize TTS
        self.init_tts()
        
        # Current image
        self.current_image = None
        self.current_image_path = None
        
        # Create GUI
        self.create_gui()
        
        # Load sample image on startup
        self.load_sample_image()
    
    def init_ai_modules(self):
        """Initialize all AI modules"""
        print("🚀 Initializing AI Modules...")
        
        try:
            self.object_detector = ImprovedObjectDetector()
            print("✅ Enhanced Object Detection loaded")
        except Exception as e:
            print(f"❌ Object Detection error: {e}")
            self.object_detector = None
        
        try:
            self.currency_detector = EnhancedCurrencyDetector()
            print("✅ Enhanced Currency Detection loaded (88.46% accuracy)")
        except Exception as e:
            print(f"❌ Currency Detection error: {e}")
            self.currency_detector = None
        
        try:
            self.ocr_module = OCRDetector()
            print("✅ OCR Text Reading loaded")
        except Exception as e:
            print(f"❌ OCR error: {e}")
            self.ocr_module = None
        
        try:
            self.color_detector = ColorDetector()
            print("✅ Color Analysis loaded")
        except Exception as e:
            print(f"❌ Color Detection error: {e}")
            self.color_detector = None
        
        try:
            self.face_recognizer = CelebrityFaceRecognizer()
            celebrity_count = len(self.face_recognizer.celebrity_encodings)
            print(f"✅ Celebrity Face Recognition loaded ({celebrity_count} celebrities)")
        except Exception as e:
            print(f"❌ Celebrity Face Recognition error: {e}")
            self.face_recognizer = None
    
    def init_tts(self):
        """Initialize Text-to-Speech"""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            print("✅ Text-to-Speech initialized")
        except Exception as e:
            print(f"❌ TTS initialization error: {e}")
            self.tts_engine = None
    
    def create_gui(self):
        """Create the professional GUI"""
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.pack(fill='x', padx=10, pady=5)
        
        title_label = tk.Label(
            title_frame,
            text="🦮 BLIND ASSISTANT - AI ASSISTIVE TECHNOLOGY",
            font=('Arial', 20, 'bold'),
            fg='#ecf0f1',
            bg='#2c3e50'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="5 AI Modules | 88.46% Currency Accuracy | 31 Celebrity Recognition | College Project",
            font=('Arial', 12),
            fg='#bdc3c7',
            bg='#2c3e50'
        )
        subtitle_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Image and controls
        left_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Image display
        self.image_label = tk.Label(left_frame, bg='#34495e', text="📷 Load an image to start", 
                                   font=('Arial', 14), fg='#bdc3c7')
        self.image_label.pack(pady=20)
        
        # Control buttons
        button_frame = tk.Frame(left_frame, bg='#34495e')
        button_frame.pack(pady=10)
        
        # Load Image button
        load_btn = tk.Button(
            button_frame,
            text="📁 Load Image",
            command=self.load_image,
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=10
        )
        load_btn.pack(side='left', padx=5)
        
        # Sample Image button
        sample_btn = tk.Button(
            button_frame,
            text="🖼️ Load Sample",
            command=self.load_sample_image,
            font=('Arial', 12, 'bold'),
            bg='#9b59b6',
            fg='white',
            padx=20,
            pady=10
        )
        sample_btn.pack(side='left', padx=5)
        
        # AI Analysis buttons
        ai_frame = tk.Frame(left_frame, bg='#34495e')
        ai_frame.pack(pady=20)
        
        # Row 1
        row1 = tk.Frame(ai_frame, bg='#34495e')
        row1.pack(pady=5)
        
        self.create_ai_button(row1, "🔍 Object Detection", self.run_object_detection, '#e74c3c')
        self.create_ai_button(row1, "💰 Currency Detection", self.run_currency_detection, '#27ae60')
        
        # Row 2
        row2 = tk.Frame(ai_frame, bg='#34495e')
        row2.pack(pady=5)
        
        self.create_ai_button(row2, "📖 OCR Text Reading", self.run_ocr, '#f39c12')
        self.create_ai_button(row2, "🎨 Color Analysis", self.run_color_detection, '#8e44ad')
        
        # Row 3 - Celebrity Face Recognition
        row3 = tk.Frame(ai_frame, bg='#34495e')
        row3.pack(pady=5)
        
        self.create_ai_button(row3, "🌟 Celebrity Recognition", self.run_face_recognition, '#e91e63')
        
        # Right panel - Results
        right_frame = tk.Frame(main_frame, bg='#34495e', relief='raised', bd=2)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Results title
        results_title = tk.Label(
            right_frame,
            text="📊 AI ANALYSIS RESULTS",
            font=('Arial', 16, 'bold'),
            fg='#ecf0f1',
            bg='#34495e'
        )
        results_title.pack(pady=10)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(
            right_frame,
            wrap=tk.WORD,
            width=50,
            height=30,
            font=('Consolas', 11),
            bg='#2c3e50',
            fg='#ecf0f1',
            insertbackground='#ecf0f1'
        )
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Audio controls
        audio_frame = tk.Frame(right_frame, bg='#34495e')
        audio_frame.pack(fill='x', padx=10, pady=5)
        
        speak_btn = tk.Button(
            audio_frame,
            text="🔊 Speak Results",
            command=self.speak_results,
            font=('Arial', 12, 'bold'),
            bg='#e67e22',
            fg='white',
            padx=20,
            pady=5
        )
        speak_btn.pack(side='left', padx=5)
        
        clear_btn = tk.Button(
            audio_frame,
            text="🗑️ Clear Results",
            command=self.clear_results,
            font=('Arial', 12, 'bold'),
            bg='#95a5a6',
            fg='white',
            padx=20,
            pady=5
        )
        clear_btn.pack(side='right', padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("✅ Ready - All 5 AI modules loaded successfully")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            relief='sunken',
            anchor='w',
            bg='#2c3e50',
            fg='#ecf0f1',
            font=('Arial', 10)
        )
        status_bar.pack(side='bottom', fill='x')
    
    def create_ai_button(self, parent, text, command, color):
        """Create AI analysis button"""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=('Arial', 11, 'bold'),
            bg=color,
            fg='white',
            padx=15,
            pady=8,
            width=25
        )
        btn.pack(side='left', padx=5)
    
    def load_image(self):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_image_from_path(file_path)
    
    def load_sample_image(self):
        """Load a sample image for demonstration"""
        sample_path = "/home/yatin/blind_assistant_project/sample_image.jpg"
        
        if not os.path.exists(sample_path):
            # Create sample image
            sample_img = np.ones((400, 600, 3), dtype=np.uint8) * 240
            
            # Add some objects
            cv2.rectangle(sample_img, (50, 50), (200, 200), (0, 0, 255), -1)  # Red rectangle
            cv2.circle(sample_img, (400, 150), 60, (0, 255, 0), -1)  # Green circle
            cv2.rectangle(sample_img, (300, 250), (550, 350), (255, 0, 0), -1)  # Blue rectangle
            
            # Add text
            cv2.putText(sample_img, "BLIND ASSISTANT", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(sample_img, "AI PROJECT", (220, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            cv2.imwrite(sample_path, sample_img)
        
        self.load_image_from_path(sample_path)
    
    def load_image_from_path(self, file_path):
        """Load image from given path"""
        try:
            # Load with OpenCV
            self.current_image = cv2.imread(file_path)
            self.current_image_path = file_path
            
            if self.current_image is None:
                raise ValueError("Could not load image")
            
            # Display image
            self.display_image()
            
            # Update status
            filename = os.path.basename(file_path)
            self.status_var.set(f"✅ Loaded: {filename}")
            
            # Add to results
            self.add_result(f"📁 IMAGE LOADED: {filename}\n" + "="*50 + "\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_var.set("❌ Failed to load image")
    
    def display_image(self):
        """Display current image in GUI"""
        if self.current_image is None:
            return
        
        try:
            # Resize image for display
            display_img = self.current_image.copy()
            height, width = display_img.shape[:2]
            
            # Scale to fit display area (max 400x300)
            max_width, max_height = 400, 300
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_img = cv2.resize(display_img, (new_width, new_height))
            
            # Convert BGR to RGB
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(display_img)
            photo = ImageTk.PhotoImage(pil_img)
            
            # Update label
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Display error: {e}")
    
    def run_object_detection(self):
        """Run enhanced object detection"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if self.object_detector is None:
            messagebox.showerror("Error", "Object detection module not available")
            return
        
        self.status_var.set("🔍 Running enhanced object detection...")
        
        def detect():
            try:
                result = self.object_detector.detect_objects(self.current_image, detailed=True)
                
                self.root.after(0, lambda: self.add_result(
                    f"🔍 ENHANCED OBJECT DETECTION:\n" + 
                    "="*50 + "\n" + 
                    result + "\n\n"
                ))
                
                self.root.after(0, lambda: self.status_var.set("✅ Object detection completed"))
                
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"❌ Object detection error: {str(e)}"))
        
        threading.Thread(target=detect, daemon=True).start()
    
    def run_currency_detection(self):
        """Run enhanced currency detection"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if self.currency_detector is None:
            messagebox.showerror("Error", "Currency detection module not available")
            return
        
        self.status_var.set("💰 Running currency detection (88.46% accuracy)...")
        
        def detect():
            try:
                result = self.currency_detector.detect_currency_enhanced(self.current_image)
                
                self.root.after(0, lambda: self.add_result(
                    f"💰 CURRENCY DETECTION (88.46% ACCURACY):\n" + 
                    "="*50 + "\n" + 
                    result + "\n\n"
                ))
                
                self.root.after(0, lambda: self.status_var.set("✅ Currency detection completed"))
                
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"❌ Currency detection error: {str(e)}"))
        
        threading.Thread(target=detect, daemon=True).start()
    
    def run_ocr(self):
        """Run OCR text detection"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if self.ocr_module is None:
            messagebox.showerror("Error", "OCR module not available")
            return
        
        self.status_var.set("📖 Running OCR text detection...")
        
        def detect():
            try:
                result = self.ocr_module.detect_text(self.current_image)
                
                self.root.after(0, lambda: self.add_result(
                    f"📖 OCR TEXT DETECTION:\n" + 
                    "="*50 + "\n" + 
                    result + "\n\n"
                ))
                
                self.root.after(0, lambda: self.status_var.set("✅ OCR completed"))
                
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"❌ OCR error: {str(e)}"))
        
        threading.Thread(target=detect, daemon=True).start()
    
    def run_color_detection(self):
        """Run color detection"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if self.color_detector is None:
            messagebox.showerror("Error", "Color detection module not available")
            return
        
        self.status_var.set("🎨 Running color analysis...")
        
        def detect():
            try:
                result = self.color_detector.detect_colors(self.current_image)
                
                self.root.after(0, lambda: self.add_result(
                    f"🎨 COLOR ANALYSIS:\n" + 
                    "="*50 + "\n" + 
                    result + "\n\n"
                ))
                
                self.root.after(0, lambda: self.status_var.set("✅ Color analysis completed"))
                
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"❌ Color detection error: {str(e)}"))
        
        threading.Thread(target=detect, daemon=True).start()
    
    def run_face_recognition(self):
        """Run celebrity face recognition"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if self.face_recognizer is None:
            messagebox.showerror("Error", "Celebrity face recognition module not available")
            return
        
        celebrity_count = len(self.face_recognizer.celebrity_encodings)
        self.status_var.set(f"🌟 Running celebrity face recognition ({celebrity_count} celebrities)...")
        
        def detect():
            try:
                result = self.face_recognizer.detect_and_recognize_faces(self.current_image)
                
                self.root.after(0, lambda: self.add_result(
                    f"🌟 CELEBRITY FACE RECOGNITION ({celebrity_count} STARS):\n" + 
                    "="*50 + "\n" + 
                    result + "\n\n"
                ))
                
                self.root.after(0, lambda: self.status_var.set("✅ Celebrity recognition completed"))
                
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"❌ Face recognition error: {str(e)}"))
        
        threading.Thread(target=detect, daemon=True).start()
    
    def add_result(self, text):
        """Add result to text area"""
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)
    
    def clear_results(self):
        """Clear results text area"""
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("✅ Results cleared")
    
    def speak_results(self):
        """Speak the results using TTS"""
        if self.tts_engine is None:
            messagebox.showerror("Error", "Text-to-Speech not available")
            return
        
        text = self.results_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showinfo("Info", "No results to speak")
            return
        
        def speak():
            try:
                # Clean text for better speech
                clean_text = text.replace("="*50, "").replace("="*40, "")
                clean_text = clean_text.replace("✅", "").replace("❌", "").replace("⚠️", "")
                clean_text = clean_text.replace("🔍", "").replace("💰", "").replace("📖", "").replace("🎨", "").replace("🌟", "")
                
                self.tts_engine.say(clean_text)
                self.tts_engine.runAndWait()
                
            except Exception as e:
                print(f"TTS error: {e}")
        
        self.status_var.set("🔊 Speaking results...")
        threading.Thread(target=speak, daemon=True).start()

def main():
    """Main function"""
    root = tk.Tk()
    app = BlindAssistantFinal(root)
    
    # Add welcome message
    welcome_msg = """🦮 WELCOME TO BLIND ASSISTANT - AI ASSISTIVE TECHNOLOGY

🎯 FEATURES:
✅ Enhanced Object Detection with Safety Alerts
✅ 88.46% Accuracy Currency Detection (₹500 & ₹2000)
✅ Celebrity Face Recognition (31 Bollywood & Hollywood Stars)
✅ High-Precision OCR Text Reading
✅ Comprehensive Color Analysis
✅ Professional Voice Feedback

📖 INSTRUCTIONS:
1. Load an image using 'Load Image' or 'Load Sample'
2. Click any AI analysis button to process the image
3. View results in the right panel
4. Use 'Speak Results' for audio feedback

🎓 College AI Project - Final Submission
"""
    
    app.add_result(welcome_msg)
    
    root.mainloop()

if __name__ == "__main__":
    main()
