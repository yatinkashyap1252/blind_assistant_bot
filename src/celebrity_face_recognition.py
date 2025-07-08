import cv2
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import glob
import dlib
import face_recognition

class CelebrityFaceRecognizer:
    def __init__(self):
        self.dataset_path = "/home/yatin/blind_assistant_project/datasets/faces/Faces/Faces"
        self.model_path = "/home/yatin/blind_assistant_project/models/custom/celebrity_face_model.pkl"
        self.encodings_path = "/home/yatin/blind_assistant_project/models/custom/celebrity_encodings.pkl"
        
        # Celebrity information database
        self.celebrity_info = {
            'Akshay Kumar': {
                'profession': 'Bollywood Actor',
                'nationality': 'Indian',
                'known_for': 'Action and Comedy films',
                'category': 'Bollywood'
            },
            'Alia Bhatt': {
                'profession': 'Bollywood Actress',
                'nationality': 'Indian',
                'known_for': 'Drama and Romance films',
                'category': 'Bollywood'
            },
            'Amitabh Bachchan': {
                'profession': 'Bollywood Actor',
                'nationality': 'Indian',
                'known_for': 'Legendary Bollywood Actor',
                'category': 'Bollywood'
            },
            'Anushka Sharma': {
                'profession': 'Bollywood Actress',
                'nationality': 'Indian',
                'known_for': 'Drama and Romance films',
                'category': 'Bollywood'
            },
            'Hrithik Roshan': {
                'profession': 'Bollywood Actor',
                'nationality': 'Indian',
                'known_for': 'Action and Dance films',
                'category': 'Bollywood'
            },
            'Priyanka Chopra': {
                'profession': 'Actress & Singer',
                'nationality': 'Indian',
                'known_for': 'Bollywood and Hollywood films',
                'category': 'International'
            },
            'Vijay Deverakonda': {
                'profession': 'Telugu Actor',
                'nationality': 'Indian',
                'known_for': 'South Indian films',
                'category': 'Tollywood'
            },
            'Virat Kohli': {
                'profession': 'Cricketer',
                'nationality': 'Indian',
                'known_for': 'Indian Cricket Captain',
                'category': 'Sports'
            },
            'Brad Pitt': {
                'profession': 'Hollywood Actor',
                'nationality': 'American',
                'known_for': 'Drama and Action films',
                'category': 'Hollywood'
            },
            'Dwayne Johnson': {
                'profession': 'Actor & Wrestler',
                'nationality': 'American',
                'known_for': 'Action films and WWE',
                'category': 'Hollywood'
            },
            'Henry Cavill': {
                'profession': 'Hollywood Actor',
                'nationality': 'British',
                'known_for': 'Superman and Action films',
                'category': 'Hollywood'
            },
            'Hugh Jackman': {
                'profession': 'Hollywood Actor',
                'nationality': 'Australian',
                'known_for': 'Wolverine and Musicals',
                'category': 'Hollywood'
            },
            'Robert Downey Jr': {
                'profession': 'Hollywood Actor',
                'nationality': 'American',
                'known_for': 'Iron Man and Marvel films',
                'category': 'Hollywood'
            },
            'Tom Cruise': {
                'profession': 'Hollywood Actor',
                'nationality': 'American',
                'known_for': 'Mission Impossible and Action films',
                'category': 'Hollywood'
            },
            'Zac Efron': {
                'profession': 'Hollywood Actor',
                'nationality': 'American',
                'known_for': 'High School Musical and Romance films',
                'category': 'Hollywood'
            },
            'Alexandra Daddario': {
                'profession': 'Hollywood Actress',
                'nationality': 'American',
                'known_for': 'Percy Jackson and TV series',
                'category': 'Hollywood'
            },
            'Charlize Theron': {
                'profession': 'Hollywood Actress',
                'nationality': 'South African',
                'known_for': 'Action and Drama films',
                'category': 'Hollywood'
            },
            'Elizabeth Olsen': {
                'profession': 'Hollywood Actress',
                'nationality': 'American',
                'known_for': 'Marvel films (Scarlet Witch)',
                'category': 'Hollywood'
            },
            'Jessica Alba': {
                'profession': 'Hollywood Actress',
                'nationality': 'American',
                'known_for': 'Action and Drama films',
                'category': 'Hollywood'
            },
            'Margot Robbie': {
                'profession': 'Hollywood Actress',
                'nationality': 'Australian',
                'known_for': 'Barbie and Drama films',
                'category': 'Hollywood'
            },
            'Natalie Portman': {
                'profession': 'Hollywood Actress',
                'nationality': 'American',
                'known_for': 'Black Swan and Drama films',
                'category': 'Hollywood'
            },
            'Billie Eilish': {
                'profession': 'Singer',
                'nationality': 'American',
                'known_for': 'Pop Music and Grammy Awards',
                'category': 'Music'
            },
            'Camila Cabello': {
                'profession': 'Singer',
                'nationality': 'Cuban-American',
                'known_for': 'Pop Music and Solo career',
                'category': 'Music'
            },
            'Roger Federer': {
                'profession': 'Tennis Player',
                'nationality': 'Swiss',
                'known_for': 'Tennis Grand Slams',
                'category': 'Sports'
            }
        }
        
        # Try to load existing model, otherwise create new one
        self.celebrity_encodings = {}
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing celebrity encodings or create new ones"""
        try:
            if os.path.exists(self.encodings_path):
                with open(self.encodings_path, 'rb') as f:
                    self.celebrity_encodings = pickle.load(f)
                print(f"✅ Loaded celebrity encodings for {len(self.celebrity_encodings)} celebrities")
            else:
                print("🔄 Creating new celebrity face encodings...")
                self.create_celebrity_encodings()
        except Exception as e:
            print(f"❌ Error loading celebrity model: {e}")
            print("🔄 Creating new celebrity face encodings...")
            self.create_celebrity_encodings()
    
    def create_celebrity_encodings(self):
        """Create face encodings for all celebrities in the dataset"""
        try:
            if not os.path.exists(self.dataset_path):
                print(f"❌ Dataset path not found: {self.dataset_path}")
                return False
            
            print("🔄 Processing celebrity images...")
            celebrity_encodings = {}
            processed_count = 0
            
            # Get all image files
            image_files = glob.glob(os.path.join(self.dataset_path, "*.jpg"))
            image_files = [f for f in image_files if "Zone.Identifier" not in f]
            
            print(f"📊 Found {len(image_files)} celebrity images to process")
            
            for image_path in image_files:
                try:
                    # Extract celebrity name from filename
                    filename = os.path.basename(image_path)
                    celebrity_name = filename.split('_')[0]
                    
                    # Load and process image
                    image = face_recognition.load_image_file(image_path)
                    
                    # Get face encodings
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if len(face_encodings) > 0:
                        # Use the first face found
                        encoding = face_encodings[0]
                        
                        # Store encoding
                        if celebrity_name not in celebrity_encodings:
                            celebrity_encodings[celebrity_name] = []
                        
                        celebrity_encodings[celebrity_name].append(encoding)
                        processed_count += 1
                        
                        if processed_count % 100 == 0:
                            print(f"📈 Processed {processed_count} images...")
                
                except Exception as e:
                    print(f"⚠️ Error processing {image_path}: {e}")
                    continue
            
            # Average encodings for each celebrity
            print("🔄 Computing average encodings for each celebrity...")
            for celebrity_name, encodings in celebrity_encodings.items():
                if len(encodings) > 0:
                    # Compute average encoding
                    avg_encoding = np.mean(encodings, axis=0)
                    celebrity_encodings[celebrity_name] = avg_encoding
                    print(f"✅ {celebrity_name}: {len(encodings)} images processed")
            
            self.celebrity_encodings = celebrity_encodings
            
            # Save encodings
            os.makedirs(os.path.dirname(self.encodings_path), exist_ok=True)
            with open(self.encodings_path, 'wb') as f:
                pickle.dump(self.celebrity_encodings, f)
            
            print(f"✅ Celebrity encodings created and saved!")
            print(f"📊 Total celebrities: {len(self.celebrity_encodings)}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating celebrity encodings: {e}")
            return False
    
    def detect_and_recognize_faces(self, image):
        """Detect faces and match with celebrities"""
        try:
            if len(self.celebrity_encodings) == 0:
                return "❌ Celebrity face database not available. Please ensure the dataset is properly loaded."
            
            # Convert BGR to RGB (OpenCV to face_recognition format)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if len(face_locations) == 0:
                return "👤 No faces detected in the image. Please ensure the image contains clear, visible faces."
            
            results = []
            
            for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                # Find best celebrity match
                best_match = self.find_best_celebrity_match(face_encoding)
                
                # Get face position
                top, right, bottom, left = face_location
                face_center_x = (left + right) // 2
                face_center_y = (top + bottom) // 2
                
                # Determine position in image
                img_height, img_width = image.shape[:2]
                position = self.get_face_position(face_center_x, face_center_y, img_width, img_height)
                
                # Calculate face size
                face_width = right - left
                face_height = bottom - top
                face_size = self.get_face_size_description(face_width * face_height, img_width * img_height)
                
                results.append({
                    'face_number': i + 1,
                    'position': position,
                    'size': face_size,
                    'celebrity_match': best_match,
                    'coordinates': face_location
                })
            
            return self.format_face_recognition_results(results)
            
        except Exception as e:
            return f"❌ Face recognition error: {str(e)}"
    
    def find_best_celebrity_match(self, face_encoding):
        """Find the best celebrity match for a face encoding"""
        try:
            best_match = None
            best_distance = float('inf')
            
            for celebrity_name, celebrity_encoding in self.celebrity_encodings.items():
                # Calculate face distance (lower is better)
                distance = face_recognition.face_distance([celebrity_encoding], face_encoding)[0]
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = celebrity_name
            
            # Convert distance to similarity percentage
            similarity = max(0, (1 - best_distance) * 100)
            
            return {
                'name': best_match,
                'similarity': similarity,
                'distance': best_distance
            }
            
        except Exception as e:
            print(f"Error finding celebrity match: {e}")
            return {
                'name': 'Unknown',
                'similarity': 0,
                'distance': 1.0
            }
    
    def get_face_position(self, center_x, center_y, img_width, img_height):
        """Get descriptive position of face in image"""
        # Horizontal position
        if center_x < img_width * 0.33:
            horizontal = "left"
        elif center_x > img_width * 0.67:
            horizontal = "right"
        else:
            horizontal = "center"
        
        # Vertical position
        if center_y < img_height * 0.33:
            vertical = "top"
        elif center_y > img_height * 0.67:
            vertical = "bottom"
        else:
            vertical = "middle"
        
        if horizontal == "center" and vertical == "middle":
            return "center of image"
        elif horizontal == "center":
            return f"{vertical} center"
        elif vertical == "middle":
            return f"{horizontal} side"
        else:
            return f"{vertical} {horizontal}"
    
    def get_face_size_description(self, face_area, total_area):
        """Get descriptive size of face"""
        size_ratio = face_area / total_area
        
        if size_ratio > 0.15:
            return "large face"
        elif size_ratio > 0.08:
            return "medium face"
        elif size_ratio > 0.03:
            return "small face"
        else:
            return "tiny face"
    
    def format_face_recognition_results(self, results):
        """Format face recognition results for display"""
        if not results:
            return "👤 No faces detected in the image."
        
        response = f"👥 CELEBRITY FACE RECOGNITION RESULTS\n"
        response += f"{'='*50}\n\n"
        response += f"🔍 Detected {len(results)} face(s):\n\n"
        
        for result in results:
            face_num = result['face_number']
            position = result['position']
            size = result['size']
            match = result['celebrity_match']
            
            response += f"👤 FACE #{face_num}:\n"
            response += f"📍 Position: {position}\n"
            response += f"📏 Size: {size}\n\n"
            
            if match['similarity'] > 60:
                celebrity_name = match['name']
                similarity = match['similarity']
                
                response += f"🌟 CELEBRITY MATCH: {celebrity_name}\n"
                response += f"📊 Similarity: {similarity:.1f}%\n"
                
                # Add celebrity information
                if celebrity_name in self.celebrity_info:
                    info = self.celebrity_info[celebrity_name]
                    response += f"🎭 Profession: {info['profession']}\n"
                    response += f"🌍 Nationality: {info['nationality']}\n"
                    response += f"⭐ Known for: {info['known_for']}\n"
                    response += f"🎬 Category: {info['category']}\n"
                
                # Confidence level
                if similarity > 80:
                    response += f"✅ High confidence match!\n"
                elif similarity > 70:
                    response += f"✅ Good match!\n"
                else:
                    response += f"⚠️ Moderate match - could be similar looking\n"
                    
            elif match['similarity'] > 40:
                response += f"🤔 POSSIBLE MATCH: {match['name']}\n"
                response += f"📊 Similarity: {match['similarity']:.1f}%\n"
                response += f"⚠️ Low confidence - might be a similar looking person\n"
            else:
                response += f"❓ NO CLEAR CELEBRITY MATCH\n"
                response += f"📊 Best similarity: {match['similarity']:.1f}% with {match['name']}\n"
                response += f"💭 This person doesn't closely match any celebrity in our database\n"
            
            response += f"\n" + "-"*40 + "\n\n"
        
        # Add summary
        high_confidence_matches = [r for r in results if r['celebrity_match']['similarity'] > 70]
        if high_confidence_matches:
            response += f"🎯 SUMMARY:\n"
            response += f"✅ {len(high_confidence_matches)} high-confidence celebrity match(es) found\n"
            
            celebrities_found = [r['celebrity_match']['name'] for r in high_confidence_matches]
            response += f"🌟 Celebrities detected: {', '.join(celebrities_found)}\n"
        
        return response
    
    def get_celebrity_stats(self):
        """Get statistics about the celebrity database"""
        if not self.celebrity_encodings:
            return "❌ Celebrity database not loaded"
        
        stats = f"📊 CELEBRITY DATABASE STATISTICS\n"
        stats += f"{'='*40}\n"
        stats += f"👥 Total Celebrities: {len(self.celebrity_encodings)}\n\n"
        
        # Categorize celebrities
        categories = {}
        for name, info in self.celebrity_info.items():
            if name in self.celebrity_encodings:
                category = info['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append(name)
        
        stats += f"📂 Categories:\n"
        for category, celebrities in categories.items():
            stats += f"  🎬 {category}: {len(celebrities)} celebrities\n"
        
        return stats
