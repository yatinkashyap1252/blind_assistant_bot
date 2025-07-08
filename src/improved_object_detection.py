import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os

class ImprovedObjectDetector:
    def __init__(self):
        try:
            # Use YOLOv8s for better accuracy (small model, good balance)
            self.model = YOLO('yolov8s.pt')  # Better than nano version
            self.class_names = self.model.names
            
            # Enhanced object categories for blind assistance
            self.safety_critical = {
                'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
                'traffic light', 'stop sign', 'fire hydrant'
            }
            
            self.navigation_aids = {
                'door', 'chair', 'couch', 'bed', 'dining table', 'toilet',
                'tv', 'laptop', 'stairs', 'handbag', 'suitcase'
            }
            
            self.daily_objects = {
                'bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                'cell phone', 'book', 'clock', 'scissors', 'teddy bear'
            }
            
            # Confidence thresholds for different object types
            self.confidence_thresholds = {
                'person': 0.3,
                'car': 0.4,
                'truck': 0.4,
                'bus': 0.4,
                'motorcycle': 0.4,
                'bicycle': 0.4,
                'default': 0.5
            }
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
    
    def detect_objects(self, image, detailed=True):
        """Enhanced object detection with better accuracy"""
        if self.model is None:
            return "Object detection model not available"
        
        try:
            # Preprocess image for better detection
            processed_image = self.preprocess_image(image)
            
            # Run inference with optimized parameters
            results = self.model(
                processed_image,
                conf=0.25,  # Lower confidence threshold for initial detection
                iou=0.45,   # Non-maximum suppression threshold
                verbose=False,
                device='cpu'  # Explicitly use CPU
            )
            
            if not results or len(results) == 0:
                return "No objects detected. Try adjusting the camera angle or lighting."
            
            # Process and filter results
            detections = self.process_detections(results[0], image.shape)
            
            if not detections:
                return "No clear objects detected. Ensure good lighting and focus."
            
            # Generate response
            return self.generate_response(detections, detailed)
            
        except Exception as e:
            return f"Object detection error: {str(e)}"
    
    def preprocess_image(self, image):
        """Preprocess image for better YOLO detection"""
        try:
            # Resize if image is too large (YOLO works best with certain sizes)
            height, width = image.shape[:2]
            if max(height, width) > 1280:
                scale = 1280 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # Enhance contrast and brightness
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return image
    
    def process_detections(self, result, image_shape):
        """Process YOLO detections with smart filtering"""
        detections = []
        
        if result.boxes is None:
            return detections
        
        height, width = image_shape[:2]
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.class_names[class_id]
            
            # Apply dynamic confidence thresholds
            threshold = self.confidence_thresholds.get(class_name, 
                                                     self.confidence_thresholds['default'])
            
            if confidence >= threshold:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Calculate object properties
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                object_width = x2 - x1
                object_height = y2 - y1
                object_area = object_width * object_height
                
                # Enhanced position description
                position = self.get_detailed_position(center_x, center_y, width, height)
                
                # Improved distance estimation
                distance = self.estimate_distance_v2(class_name, object_area, width * height, 
                                                   object_width, object_height)
                
                # Object size description
                size_desc = self.get_size_description(object_area, width * height)
                
                detection_info = {
                    'object': class_name,
                    'confidence': confidence,
                    'position': position,
                    'distance': distance,
                    'size': size_desc,
                    'coordinates': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'area': object_area
                }
                
                detections.append(detection_info)
        
        # Sort by importance (safety first, then confidence)
        detections.sort(key=lambda x: (
            x['object'] in self.safety_critical,
            x['confidence']
        ), reverse=True)
        
        return detections
    
    def get_detailed_position(self, center_x, center_y, img_width, img_height):
        """Enhanced position description with more precision"""
        # Divide image into 9 zones (3x3 grid)
        h_zone = 0 if center_x < img_width * 0.33 else (1 if center_x < img_width * 0.67 else 2)
        v_zone = 0 if center_y < img_height * 0.33 else (1 if center_y < img_height * 0.67 else 2)
        
        positions = [
            ["top-left", "top-center", "top-right"],
            ["left side", "center", "right side"],
            ["bottom-left", "bottom-center", "bottom-right"]
        ]
        
        base_position = positions[v_zone][h_zone]
        
        # Add distance from center for more precision
        center_dist_x = abs(center_x - img_width/2) / (img_width/2)
        center_dist_y = abs(center_y - img_height/2) / (img_height/2)
        
        if center_dist_x < 0.1 and center_dist_y < 0.1:
            return "directly ahead"
        elif base_position == "center":
            return "center of view"
        else:
            return base_position
    
    def estimate_distance_v2(self, object_name, object_area, total_area, obj_width, obj_height):
        """Improved distance estimation using multiple factors"""
        size_ratio = object_area / total_area
        
        # Object-specific size expectations (typical pixel areas at different distances)
        size_expectations = {
            'person': {'very_close': 0.4, 'close': 0.2, 'medium': 0.08, 'far': 0.03},
            'car': {'very_close': 0.5, 'close': 0.25, 'medium': 0.1, 'far': 0.04},
            'chair': {'very_close': 0.3, 'close': 0.15, 'medium': 0.06, 'far': 0.02},
            'bottle': {'very_close': 0.15, 'close': 0.08, 'medium': 0.03, 'far': 0.01},
            'cell phone': {'very_close': 0.1, 'close': 0.05, 'medium': 0.02, 'far': 0.008}
        }
        
        # Default expectations for unknown objects
        expectations = size_expectations.get(object_name, {
            'very_close': 0.3, 'close': 0.15, 'medium': 0.06, 'far': 0.02
        })
        
        if size_ratio >= expectations['very_close']:
            return "very close"
        elif size_ratio >= expectations['close']:
            return "close"
        elif size_ratio >= expectations['medium']:
            return "medium distance"
        elif size_ratio >= expectations['far']:
            return "far"
        else:
            return "very far"
    
    def get_size_description(self, object_area, total_area):
        """Describe object size relative to image"""
        size_ratio = object_area / total_area
        
        if size_ratio > 0.4:
            return "large"
        elif size_ratio > 0.15:
            return "medium"
        elif size_ratio > 0.05:
            return "small"
        else:
            return "tiny"
    
    def generate_response(self, detections, detailed=True):
        """Generate comprehensive response for blind users"""
        if not detections:
            return "No objects detected clearly."
        
        response = f"🔍 Detected {len(detections)} object(s):\n\n"
        
        # Safety alerts first
        safety_alerts = []
        for detection in detections:
            obj_name = detection['object']
            distance = detection['distance']
            position = detection['position']
            
            if obj_name in self.safety_critical:
                if obj_name in ['car', 'truck', 'bus', 'motorcycle'] and distance in ["very close", "close"]:
                    safety_alerts.append(f"⚠️ CAUTION: {obj_name} detected {distance} on your {position}!")
                elif obj_name == 'person' and distance in ["very close", "close"]:
                    safety_alerts.append(f"👤 Person detected {distance} - {position}")
        
        if safety_alerts:
            response += "🚨 SAFETY ALERTS:\n"
            for alert in safety_alerts:
                response += f"{alert}\n"
            response += "\n"
        
        # Main object descriptions (limit to top 6 for clarity)
        response += "📍 OBJECTS DETECTED:\n"
        for i, detection in enumerate(detections[:6], 1):
            obj_name = detection['object'].replace('_', ' ').title()
            position = detection['position']
            distance = detection['distance']
            size = detection['size']
            confidence = detection['confidence']
            
            # Categorize object
            category = ""
            if detection['object'] in self.safety_critical:
                category = "⚠️ "
            elif detection['object'] in self.navigation_aids:
                category = "🧭 "
            elif detection['object'] in self.daily_objects:
                category = "🏠 "
            
            if detailed:
                response += f"{i}. {category}{obj_name}\n"
                response += f"   📍 Position: {position}\n"
                response += f"   📏 Distance: {distance}\n"
                response += f"   📐 Size: {size}\n"
                response += f"   ✅ Confidence: {confidence:.0%}\n\n"
            else:
                response += f"{i}. {category}{obj_name} - {position}, {distance}\n"
        
        # Summary for navigation
        nav_objects = [d for d in detections if d['object'] in self.navigation_aids]
        if nav_objects:
            response += f"\n🧭 Navigation aids: {len(nav_objects)} helpful objects detected\n"
        
        # Environment summary
        total_people = len([d for d in detections if d['object'] == 'person'])
        total_vehicles = len([d for d in detections if d['object'] in ['car', 'truck', 'bus', 'motorcycle']])
        
        if total_people > 0:
            response += f"👥 People in area: {total_people}\n"
        if total_vehicles > 0:
            response += f"🚗 Vehicles nearby: {total_vehicles}\n"
        
        return response
    
    def train_custom_model(self, dataset_path):
        """Train custom YOLO model for specific objects"""
        try:
            print("Training custom YOLO model...")
            
            # This would require a properly formatted YOLO dataset
            # For now, we'll use transfer learning with the pre-trained model
            model = YOLO('yolov8s.pt')
            
            # Train on custom dataset (if available)
            if os.path.exists(dataset_path):
                results = model.train(
                    data=dataset_path,
                    epochs=50,
                    imgsz=640,
                    batch=8,
                    device='cpu'
                )
                
                # Save the trained model
                model.save('/home/yatin/blind_assistant_project/models/custom/custom_yolo.pt')
                print("Custom model trained successfully!")
                return True
            else:
                print("Dataset path not found. Using pre-trained model.")
                return False
                
        except Exception as e:
            print(f"Training error: {e}")
            return False
