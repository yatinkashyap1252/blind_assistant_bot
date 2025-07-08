import cv2
import pytesseract
import numpy as np
from PIL import Image

class OCRDetector:
    def __init__(self):
        # Configure tesseract path if needed (uncomment and modify if required)
        # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        pass
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def detect_text(self, image):
        """Detect and extract text from image"""
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Use pytesseract to extract text
            # Configure OCR settings for better accuracy
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?-'
            
            # Extract text
            text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            # Clean up the text
            text = text.strip()
            
            if text:
                # Get confidence scores
                data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                result = f"Text detected (Confidence: {avg_confidence:.1f}%):\n\"{text}\""
                
                # Add reading guidance
                if len(text.split()) > 10:
                    result += "\n\nThis appears to be a longer text. Listen carefully."
                elif any(char.isdigit() for char in text):
                    result += "\n\nNumbers detected in the text."
                
                return result
            else:
                return "No clear text detected. Try adjusting the camera angle or lighting."
                
        except Exception as e:
            return f"OCR Error: Unable to process image. {str(e)}"
    
    def detect_text_regions(self, image):
        """Detect text regions in the image"""
        try:
            processed_image = self.preprocess_image(image)
            
            # Get bounding boxes of text regions
            data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
            
            text_regions = []
            n_boxes = len(data['level'])
            
            for i in range(n_boxes):
                if int(data['conf'][i]) > 30:  # Only consider high confidence detections
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    text = data['text'][i].strip()
                    if text:
                        text_regions.append({
                            'text': text,
                            'bbox': (x, y, w, h),
                            'confidence': data['conf'][i]
                        })
            
            return text_regions
            
        except Exception as e:
            print(f"Text region detection error: {e}")
            return []
