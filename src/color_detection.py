import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors

class ColorDetector:
    def __init__(self):
        # Define color ranges in HSV for basic color detection
        self.color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
            'orange': [(10, 50, 50), (25, 255, 255)],
            'yellow': [(25, 50, 50), (35, 255, 255)],
            'green': [(35, 50, 50), (85, 255, 255)],
            'blue': [(85, 50, 50), (125, 255, 255)],
            'purple': [(125, 50, 50), (155, 255, 255)],
            'pink': [(155, 50, 50), (170, 255, 255)],
            'white': [(0, 0, 200), (180, 30, 255)],
            'black': [(0, 0, 0), (180, 255, 30)],
            'gray': [(0, 0, 30), (180, 30, 200)]
        }
    
    def detect_colors(self, image):
        """Detect and describe colors in the image"""
        try:
            # Get dominant colors
            dominant_colors = self.get_dominant_colors(image, k=5)
            
            # Get color distribution
            color_distribution = self.analyze_color_distribution(image)
            
            # Get color names
            color_names = self.get_color_names(dominant_colors)
            
            # Analyze image properties
            brightness = self.analyze_brightness(image)
            contrast = self.analyze_contrast(image)
            
            # Format result
            result = "Color Analysis:\n\n"
            
            # Dominant colors
            result += "🎨 Dominant Colors:\n"
            for i, (color, name) in enumerate(zip(dominant_colors, color_names), 1):
                b, g, r = color.astype(int)
                result += f"{i}. {name} (RGB: {r}, {g}, {b})\n"
            
            # Color distribution
            result += f"\n📊 Color Distribution:\n"
            for color_name, percentage in color_distribution.items():
                if percentage > 5:  # Only show colors that make up more than 5%
                    result += f"• {color_name}: {percentage:.1f}%\n"
            
            # Image properties
            result += f"\n💡 Image Properties:\n"
            result += f"• Overall brightness: {brightness}\n"
            result += f"• Contrast level: {contrast}\n"
            
            # Color accessibility info
            result += self.get_accessibility_info(dominant_colors)
            
            # Practical description
            result += f"\n🗣️ Simple Description:\n"
            result += self.get_simple_description(color_names, color_distribution)
            
            return result
            
        except Exception as e:
            return f"Color detection error: {str(e)}"
    
    def get_dominant_colors(self, image, k=5):
        """Get dominant colors using K-means clustering"""
        try:
            # Reshape image to be a list of pixels
            pixels = image.reshape(-1, 3)
            
            # Remove very dark and very bright pixels for better color detection
            mask = np.logical_and(
                np.sum(pixels, axis=1) > 30,   # Not too dark
                np.sum(pixels, axis=1) < 720   # Not too bright
            )
            filtered_pixels = pixels[mask]
            
            if len(filtered_pixels) < k:
                filtered_pixels = pixels
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(filtered_pixels)
            
            # Get cluster centers (dominant colors)
            colors = kmeans.cluster_centers_
            
            # Sort by cluster size (most dominant first)
            labels = kmeans.labels_
            label_counts = np.bincount(labels)
            sorted_indices = np.argsort(label_counts)[::-1]
            
            return colors[sorted_indices]
            
        except Exception as e:
            print(f"Dominant color extraction error: {e}")
            # Fallback: return average colors from different regions
            h, w = image.shape[:2]
            regions = [
                image[0:h//2, 0:w//2],      # Top-left
                image[0:h//2, w//2:w],      # Top-right
                image[h//2:h, 0:w//2],      # Bottom-left
                image[h//2:h, w//2:w],      # Bottom-right
                image[h//4:3*h//4, w//4:3*w//4]  # Center
            ]
            return np.array([np.mean(region.reshape(-1, 3), axis=0) for region in regions])
    
    def get_color_names(self, colors):
        """Convert RGB colors to color names"""
        color_names = []
        
        for color in colors:
            try:
                # Convert BGR to RGB
                b, g, r = color.astype(int)
                
                # Try to get exact color name
                try:
                    color_name = webcolors.rgb_to_name((r, g, b))
                except ValueError:
                    # If exact match not found, find closest color
                    color_name = self.closest_color_name((r, g, b))
                
                color_names.append(color_name.title())
                
            except Exception as e:
                # Fallback to basic color detection
                color_names.append(self.basic_color_name(color))
        
        return color_names
    
    def closest_color_name(self, rgb_color):
        """Find the closest color name for an RGB value"""
        try:
            min_colors = {}
            for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
                r_c, g_c, b_c = webcolors.hex_to_rgb(key)
                rd = (r_c - rgb_color[0]) ** 2
                gd = (g_c - rgb_color[1]) ** 2
                bd = (b_c - rgb_color[2]) ** 2
                min_colors[(rd + gd + bd)] = name
            return min_colors[min(min_colors.keys())]
        except:
            return self.basic_color_name(rgb_color)
    
    def basic_color_name(self, color):
        """Basic color name detection using HSV ranges"""
        try:
            # Convert BGR to HSV
            bgr_color = np.uint8([[color]])
            hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = hsv_color
            
            # Check against predefined ranges
            for color_name, ranges in self.color_ranges.items():
                if color_name == 'red':
                    # Red has two ranges due to hue wraparound
                    if ((ranges[0][0] <= h <= ranges[1][0]) or (ranges[2][0] <= h <= ranges[3][0])) and \
                       (ranges[0][1] <= s <= ranges[1][1]) and (ranges[0][2] <= v <= ranges[1][2]):
                        return color_name.title()
                else:
                    if len(ranges) == 2:
                        lower, upper = ranges
                        if (lower[0] <= h <= upper[0]) and (lower[1] <= s <= upper[1]) and (lower[2] <= v <= upper[2]):
                            return color_name.title()
            
            # Fallback based on brightness
            if v < 50:
                return "Dark"
            elif v > 200 and s < 50:
                return "Light"
            else:
                return "Mixed"
                
        except Exception as e:
            return "Unknown"
    
    def analyze_color_distribution(self, image):
        """Analyze the distribution of basic colors in the image"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            total_pixels = image.shape[0] * image.shape[1]
            color_distribution = {}
            
            for color_name, ranges in self.color_ranges.items():
                if color_name == 'red':
                    # Handle red's two ranges
                    mask1 = cv2.inRange(hsv, ranges[0], ranges[1])
                    mask2 = cv2.inRange(hsv, ranges[2], ranges[3])
                    mask = cv2.bitwise_or(mask1, mask2)
                else:
                    if len(ranges) == 2:
                        mask = cv2.inRange(hsv, ranges[0], ranges[1])
                    else:
                        continue
                
                pixel_count = cv2.countNonZero(mask)
                percentage = (pixel_count / total_pixels) * 100
                color_distribution[color_name.title()] = percentage
            
            return color_distribution
            
        except Exception as e:
            print(f"Color distribution analysis error: {e}")
            return {}
    
    def analyze_brightness(self, image):
        """Analyze overall brightness of the image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            if avg_brightness < 50:
                return "Very Dark"
            elif avg_brightness < 100:
                return "Dark"
            elif avg_brightness < 150:
                return "Medium"
            elif avg_brightness < 200:
                return "Bright"
            else:
                return "Very Bright"
                
        except Exception as e:
            return "Unknown"
    
    def analyze_contrast(self, image):
        """Analyze contrast level of the image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray)
            
            if contrast < 30:
                return "Low Contrast"
            elif contrast < 60:
                return "Medium Contrast"
            else:
                return "High Contrast"
                
        except Exception as e:
            return "Unknown"
    
    def get_accessibility_info(self, colors):
        """Provide accessibility information about colors"""
        try:
            result = f"\n♿ Accessibility Notes:\n"
            
            # Check for high contrast
            if len(colors) >= 2:
                color1 = colors[0]
                color2 = colors[1]
                
                # Calculate luminance difference (simplified)
                lum1 = 0.299 * color1[2] + 0.587 * color1[1] + 0.114 * color1[0]
                lum2 = 0.299 * color2[2] + 0.587 * color2[1] + 0.114 * color2[0]
                contrast_ratio = abs(lum1 - lum2) / 255
                
                if contrast_ratio > 0.7:
                    result += "• Good contrast between main colors\n"
                else:
                    result += "• Low contrast - may be difficult to distinguish\n"
            
            # Check for colorblind-friendly combinations
            has_red_green = any('red' in str(colors).lower() or 'green' in str(colors).lower())
            if has_red_green:
                result += "• Contains red/green - may be challenging for colorblind users\n"
            
            return result
            
        except Exception as e:
            return f"\n♿ Accessibility analysis error: {str(e)}\n"
    
    def get_simple_description(self, color_names, distribution):
        """Generate a simple, natural description of the colors"""
        try:
            # Get the most prominent colors
            main_colors = []
            for color_name, percentage in distribution.items():
                if percentage > 10:  # Colors that make up more than 10%
                    main_colors.append(color_name.lower())
            
            if not main_colors:
                main_colors = [name.lower() for name in color_names[:2]]
            
            if len(main_colors) == 1:
                description = f"This image is primarily {main_colors[0]}."
            elif len(main_colors) == 2:
                description = f"This image contains mainly {main_colors[0]} and {main_colors[1]}."
            else:
                description = f"This image has multiple colors including {', '.join(main_colors[:3])}."
            
            # Add brightness context
            total_bright = sum(percentage for color, percentage in distribution.items() 
                             if color.lower() in ['white', 'yellow', 'light'])
            total_dark = sum(percentage for color, percentage in distribution.items() 
                           if color.lower() in ['black', 'dark', 'gray'])
            
            if total_bright > 30:
                description += " The image appears bright overall."
            elif total_dark > 30:
                description += " The image appears dark overall."
            
            return description
            
        except Exception as e:
            return "Color description unavailable."
