import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import glob
from sklearn.cluster import KMeans

class EnhancedCurrencyDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.currency_labels = {
            0: "500 Rupees Note",
            1: "2000 Rupees Note", 
            2: "Fake/Unknown Note"
        }
        
        # Enhanced currency characteristics
        self.currency_features = {
            0: {  # 500 Rupees
                'dominant_colors': ['purple', 'violet', 'magenta'],
                'size_ratio': 1.6,  # width/height ratio
                'security_features': ['Gandhi portrait', 'Ashoka pillar', 'Red Fort']
            },
            1: {  # 2000 Rupees
                'dominant_colors': ['pink', 'magenta', 'red'],
                'size_ratio': 1.6,
                'security_features': ['Gandhi portrait', 'Mars mission', 'Swachh Bharat logo']
            }
        }
        
        self.dataset_path = "/home/yatin/blind_assistant_project/datasets/currency"
        self.model_path = "/home/yatin/blind_assistant_project/models/custom/enhanced_currency_model.pkl"
        self.scaler_path = "/home/yatin/blind_assistant_project/models/custom/currency_scaler.pkl"
        
        # Load or train the model
        self.load_or_train_model()
    
    def extract_enhanced_features(self, image):
        """Extract comprehensive features from currency image"""
        try:
            # Standardize image size
            image = cv2.resize(image, (300, 150))  # Larger size for better feature extraction
            
            # Convert to different color spaces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            features = []
            
            # 1. Enhanced Color Features
            # BGR color histograms (more bins for better discrimination)
            for i in range(3):
                hist = cv2.calcHist([image], [i], None, [64], [0, 256])
                features.extend(hist.flatten())
            
            # HSV color histograms
            for i in range(3):
                hist = cv2.calcHist([hsv], [i], None, [64], [0, 256])
                features.extend(hist.flatten())
            
            # LAB color histograms
            for i in range(3):
                hist = cv2.calcHist([lab], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
            
            # 2. Statistical Color Features
            for channel in range(3):
                channel_data = image[:,:,channel].flatten()
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data),
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75),
                    np.min(channel_data),
                    np.max(channel_data)
                ])
            
            # 3. Texture Features
            # Local Binary Pattern approximation
            gray_normalized = gray.astype(np.float32) / 255.0
            
            # Calculate texture measures
            features.extend([
                np.mean(gray_normalized),
                np.std(gray_normalized),
                np.var(gray_normalized)
            ])
            
            # Gradient features
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features.extend([
                np.mean(gradient_magnitude),
                np.std(gradient_magnitude),
                np.max(gradient_magnitude)
            ])
            
            # 4. Edge Features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # Edge direction histogram
            edge_angles = np.arctan2(grad_y, grad_x)
            edge_hist, _ = np.histogram(edge_angles[edges > 0], bins=8, range=(-np.pi, np.pi))
            features.extend(edge_hist / (np.sum(edge_hist) + 1e-7))
            
            # 5. Dominant Color Analysis (K-means clustering)
            pixels = image.reshape(-1, 3)
            dominant_colors = self.get_dominant_colors_kmeans(pixels, k=5)
            features.extend(dominant_colors.flatten())
            
            # 6. Geometric Features
            # Aspect ratio
            h, w = image.shape[:2]
            features.append(w / h)
            
            # 7. Frequency Domain Features (DCT)
            gray_float = np.float32(gray)
            dct = cv2.dct(gray_float)
            # Take low frequency components
            dct_features = dct[:8, :8].flatten()
            features.extend(dct_features)
            
            # 8. Color Moments
            for channel in range(3):
                channel_data = image[:,:,channel]
                # First moment (mean)
                mean = np.mean(channel_data)
                # Second moment (variance)
                variance = np.var(channel_data)
                # Third moment (skewness)
                skewness = np.mean(((channel_data - mean) / np.sqrt(variance + 1e-7)) ** 3)
                features.extend([mean, variance, skewness])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Enhanced feature extraction error: {e}")
            return np.zeros(500)  # Return larger zero vector
    
    def get_dominant_colors_kmeans(self, pixels, k=5):
        """Get dominant colors using improved k-means clustering"""
        try:
            # Remove very dark and very bright pixels (likely shadows/highlights)
            filtered_pixels = pixels[
                (np.sum(pixels, axis=1) > 30) & (np.sum(pixels, axis=1) < 700)
            ]
            
            if len(filtered_pixels) < k:
                filtered_pixels = pixels
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
            kmeans.fit(filtered_pixels)
            
            # Sort colors by cluster size
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            
            # Count pixels in each cluster
            unique, counts = np.unique(labels, return_counts=True)
            sorted_indices = np.argsort(counts)[::-1]
            
            return centers[sorted_indices]
            
        except Exception as e:
            print(f"K-means clustering error: {e}")
            return np.array([[128, 128, 128]] * k)  # Return gray colors as fallback
    
    def load_dataset_enhanced(self):
        """Load and preprocess currency dataset with data augmentation"""
        features = []
        labels = []
        
        try:
            # Load 500 rupee notes
            rupees_500_path = os.path.join(self.dataset_path, "500_dataset")
            if os.path.exists(rupees_500_path):
                count_500 = 0
                for img_file in glob.glob(os.path.join(rupees_500_path, "*.jpg")):
                    if "Zone.Identifier" not in img_file:
                        image = cv2.imread(img_file)
                        if image is not None:
                            # Original image
                            feature_vector = self.extract_enhanced_features(image)
                            features.append(feature_vector)
                            labels.append(0)
                            count_500 += 1
                            
                            # Data augmentation
                            augmented_images = self.augment_image(image)
                            for aug_img in augmented_images:
                                aug_features = self.extract_enhanced_features(aug_img)
                                features.append(aug_features)
                                labels.append(0)
                                count_500 += 1
                
                print(f"Loaded {count_500} samples for 500 rupee notes")
            
            # Load 2000 rupee notes
            rupees_2000_path = os.path.join(self.dataset_path, "2000_dataset")
            if os.path.exists(rupees_2000_path):
                count_2000 = 0
                for img_file in glob.glob(os.path.join(rupees_2000_path, "*.jpg")):
                    if "Zone.Identifier" not in img_file:
                        image = cv2.imread(img_file)
                        if image is not None:
                            # Original image
                            feature_vector = self.extract_enhanced_features(image)
                            features.append(feature_vector)
                            labels.append(1)
                            count_2000 += 1
                            
                            # Data augmentation
                            augmented_images = self.augment_image(image)
                            for aug_img in augmented_images:
                                aug_features = self.extract_enhanced_features(aug_img)
                                features.append(aug_features)
                                labels.append(1)
                                count_2000 += 1
                
                print(f"Loaded {count_2000} samples for 2000 rupee notes")
            
            # Load fake notes
            fake_notes_path = os.path.join(self.dataset_path, "Fake Notes")
            if os.path.exists(fake_notes_path):
                count_fake = 0
                for img_file in glob.glob(os.path.join(fake_notes_path, "**/*.jpg"), recursive=True):
                    if "Zone.Identifier" not in img_file:
                        image = cv2.imread(img_file)
                        if image is not None:
                            feature_vector = self.extract_enhanced_features(image)
                            features.append(feature_vector)
                            labels.append(2)
                            count_fake += 1
                
                print(f"Loaded {count_fake} samples for fake notes")
            
            print(f"Total dataset size: {len(features)} samples")
            return np.array(features), np.array(labels)
            
        except Exception as e:
            print(f"Enhanced dataset loading error: {e}")
            return np.array([]), np.array([])
    
    def augment_image(self, image):
        """Apply data augmentation techniques"""
        augmented = []
        
        try:
            # Brightness adjustment
            bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
            dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
            augmented.extend([bright, dark])
            
            # Rotation (small angles)
            center = (image.shape[1]//2, image.shape[0]//2)
            for angle in [-5, 5]:
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                augmented.append(rotated)
            
            # Gaussian blur (slight)
            blurred = cv2.GaussianBlur(image, (3, 3), 0)
            augmented.append(blurred)
            
        except Exception as e:
            print(f"Augmentation error: {e}")
        
        return augmented
    
    def train_enhanced_model(self):
        """Train enhanced currency classification model"""
        try:
            features, labels = self.load_dataset_enhanced()
            
            if len(features) == 0:
                print("No training data available")
                return False
            
            # Feature scaling
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, labels, test_size=0.2, random_state=42, 
                stratify=labels if len(np.unique(labels)) > 1 else None
            )
            
            # Try multiple models and select the best
            models = {
                'RandomForest': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                'GradientBoosting': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            }
            
            best_model = None
            best_score = 0
            best_name = ""
            
            for name, model in models.items():
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                avg_score = np.mean(cv_scores)
                
                print(f"{name} CV Score: {avg_score:.3f} (+/- {np.std(cv_scores) * 2:.3f})")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    best_name = name
            
            # Train the best model
            print(f"Training best model: {best_name}")
            self.model = best_model
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Enhanced currency model accuracy: {accuracy:.2%}")
            print("\nDetailed Classification Report:")
            print(classification_report(y_test, y_pred, 
                                      target_names=list(self.currency_labels.values())))
            
            # Save model and scaler
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print("Enhanced currency model and scaler saved successfully")
            return True
            
        except Exception as e:
            print(f"Enhanced model training error: {e}")
            return False
    
    def load_or_train_model(self):
        """Load existing enhanced model or train new one"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("Enhanced currency model loaded successfully")
            else:
                print("Training new enhanced currency model...")
                self.train_enhanced_model()
        except Exception as e:
            print(f"Model loading error: {e}")
            self.train_enhanced_model()
    
    def detect_currency_enhanced(self, image):
        """Enhanced currency detection with comprehensive analysis"""
        try:
            if self.model is None or self.scaler is None:
                return "Enhanced currency detection model not available. Please retrain the model."
            
            # Preprocess image
            processed_image = self.preprocess_currency_image_v2(image)
            
            # Extract enhanced features
            features = self.extract_enhanced_features(processed_image)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get confidence and alternative predictions
            confidence = np.max(probabilities) * 100
            sorted_indices = np.argsort(probabilities)[::-1]
            
            # Get currency name
            currency_name = self.currency_labels.get(prediction, "Unknown")
            
            # Comprehensive analysis
            analysis = self.comprehensive_currency_analysis(processed_image, prediction, probabilities)
            
            # Build detailed result
            result = f"💰 ENHANCED CURRENCY DETECTION\n"
            result += f"{'='*40}\n\n"
            
            result += f"🏦 Primary Detection: {currency_name}\n"
            result += f"📊 Confidence: {confidence:.1f}%\n\n"
            
            # Show alternative predictions
            result += f"📈 All Predictions:\n"
            for i, idx in enumerate(sorted_indices):
                prob = probabilities[idx] * 100
                label = self.currency_labels[idx]
                result += f"  {i+1}. {label}: {prob:.1f}%\n"
            
            result += f"\n"
            
            # Confidence interpretation
            if confidence > 85:
                result += f"✅ Very High Confidence - Reliable detection\n"
            elif confidence > 70:
                result += f"✅ High Confidence - Good detection\n"
            elif confidence > 55:
                result += f"⚠️ Medium Confidence - Consider better lighting/angle\n"
            else:
                result += f"❌ Low Confidence - Image unclear or not a known currency\n"
            
            result += f"\n{analysis}"
            
            # Security warnings
            if prediction == 2 and confidence > 60:
                result += f"\n\n🚨 SECURITY ALERT: This appears to be a FAKE or UNKNOWN note!"
                result += f"\n⚠️ Please verify with additional security features"
            
            # Recommendations
            result += f"\n\n💡 RECOMMENDATIONS:\n"
            if confidence < 70:
                result += f"• Improve lighting conditions\n"
                result += f"• Ensure note is flat and fully visible\n"
                result += f"• Clean camera lens\n"
                result += f"• Hold camera steady\n"
            
            if prediction in [0, 1]:  # Valid currency
                expected_features = self.currency_features[prediction]
                result += f"• Expected colors: {', '.join(expected_features['dominant_colors'])}\n"
                result += f"• Look for security features: {', '.join(expected_features['security_features'])}\n"
            
            return result
            
        except Exception as e:
            return f"Enhanced currency detection error: {str(e)}"
    
    def preprocess_currency_image_v2(self, image):
        """Advanced preprocessing for currency images"""
        try:
            # Resize while maintaining aspect ratio
            height, width = image.shape[:2]
            target_width = 600
            target_height = int(height * (target_width / width))
            
            resized = cv2.resize(image, (target_width, target_height))
            
            # Advanced contrast enhancement
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE with optimized parameters
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Noise reduction
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            return denoised
            
        except Exception as e:
            print(f"Advanced preprocessing error: {e}")
            return image
    
    def comprehensive_currency_analysis(self, image, prediction, probabilities):
        """Comprehensive analysis of currency image"""
        try:
            analysis = f"🔍 DETAILED ANALYSIS:\n"
            analysis += f"{'-'*25}\n"
            
            # Color analysis
            avg_color = np.mean(image, axis=(0, 1))
            dominant_color = self.get_dominant_color_name_v2(avg_color)
            
            analysis += f"🎨 Dominant Color: {dominant_color}\n"
            
            # Size and aspect ratio analysis
            height, width = image.shape[:2]
            aspect_ratio = width / height
            analysis += f"📐 Dimensions: {width}x{height} (ratio: {aspect_ratio:.2f})\n"
            
            # Brightness and contrast analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            analysis += f"💡 Brightness: {brightness:.0f}/255\n"
            analysis += f"🔳 Contrast: {contrast:.0f}\n"
            
            # Edge density (complexity measure)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            analysis += f"📏 Edge Density: {edge_density:.3f}\n"
            
            # Color distribution analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue_std = np.std(hsv[:,:,0])
            saturation_mean = np.mean(hsv[:,:,1])
            
            analysis += f"🌈 Color Variety: {hue_std:.1f}\n"
            analysis += f"🎯 Color Intensity: {saturation_mean:.1f}\n"
            
            # Quality assessment
            analysis += f"\n📋 IMAGE QUALITY:\n"
            
            if brightness < 60:
                analysis += f"• Too dark - increase lighting\n"
            elif brightness > 200:
                analysis += f"• Too bright - reduce lighting\n"
            else:
                analysis += f"• Good brightness level\n"
            
            if contrast < 30:
                analysis += f"• Low contrast - image may be blurry\n"
            else:
                analysis += f"• Good contrast level\n"
            
            if edge_density < 0.05:
                analysis += f"• Low detail - move closer or focus better\n"
            else:
                analysis += f"• Good detail level\n"
            
            return analysis
            
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def get_dominant_color_name_v2(self, bgr_color):
        """Enhanced color name detection"""
        b, g, r = bgr_color
        
        # Convert to HSV for better color classification
        hsv_color = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        hue, sat, val = hsv_color
        
        if sat < 30:  # Low saturation
            if val > 200:
                return "White/Light Gray"
            elif val < 50:
                return "Black/Dark Gray"
            else:
                return "Gray"
        
        # Color classification based on hue
        if hue < 10 or hue > 170:
            return "Red/Pink"
        elif hue < 25:
            return "Orange"
        elif hue < 35:
            return "Yellow"
        elif hue < 85:
            return "Green"
        elif hue < 125:
            return "Blue/Cyan"
        elif hue < 150:
            return "Blue"
        else:
            return "Purple/Violet"
