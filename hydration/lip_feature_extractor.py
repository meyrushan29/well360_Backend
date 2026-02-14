"""
Advanced Feature Extraction for Lip Hydration Analysis
Extracts multi-modal features: texture, color, spatial patterns
"""
import numpy as np
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern


# ======================================================
# 1. LIP REGION DETECTION & CROPPING
# ======================================================
def detect_and_crop_lips(image):
    """
    Use MediaPipe Face Mesh to isolate lip region.
    Returns (cropped_image, success_flag, landmarks)
    """
    try:
        import mediapipe as mp
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        img_np = np.array(image)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) if img_np.shape[2] == 3 else img_np
        
        results = face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Lip landmark indices (upper + lower lip outer contour)
            # MediaPipe 468 landmarks - lips are indices 61-291
            lip_indices = [
                61, 185, 40, 39, 37, 0, 267, 269, 270, 409,  # Upper outer lip
                291, 375, 321, 405, 314, 17, 84, 181, 91, 146  # Lower outer lip
            ]
            
            # Extract lip bounding box
            h, w = img_rgb.shape[:2]
            x_coords = [landmarks.landmark[i].x * w for i in lip_indices]
            y_coords = [landmarks.landmark[i].y * h for i in lip_indices]
            
            # Add generous padding for context
            padding = 40
            x_min = max(0, int(min(x_coords)) - padding)
            x_max = min(w, int(max(x_coords)) + padding)
            y_min = max(0, int(min(y_coords)) - padding)
            y_max = min(h, int(max(y_coords)) + padding)
            
            # Crop to lip region
            cropped = image.crop((x_min, y_min, x_max, y_max))
            
            # Extract landmark coordinates for overlay
            lip_coords = [(landmarks.landmark[i].x * w - x_min, 
                          landmarks.landmark[i].y * h - y_min) 
                         for i in lip_indices]
            
            return cropped, True, lip_coords
        
        return image, False, None
        
    except ImportError:
        print("[Warning] MediaPipe not available, skipping lip detection")
        return image, False, None
    except Exception as e:
        print(f"[Warning] Lip detection failed: {e}")
        return image, False, None


# ======================================================
# 2. COLOR FEATURE EXTRACTION
# ======================================================
def extract_color_features(image):
    """
    Extract advanced color statistics from multiple color spaces.
    Returns dict of color features.
    """
    img_np = np.array(image)
    
    # Convert to different color spaces
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    
    # RGB Statistics
    rgb_mean = np.mean(img_np, axis=(0, 1))
    rgb_std = np.std(img_np, axis=(0, 1))
    
    # HSV Statistics
    hsv_mean = np.mean(img_hsv, axis=(0, 1))
    hsv_std = np.std(img_hsv, axis=(0, 1))
    
    # LAB Statistics
    lab_mean = np.mean(img_lab, axis=(0, 1))
    
    # Redness Ratio (dehydrated lips appear darker/redder)
    r_channel = img_np[:, :, 0].astype(float)
    g_channel = img_np[:, :, 1].astype(float)
    b_channel = img_np[:, :, 2].astype(float)
    
    # Calculate redness index
    redness = (r_channel - (g_channel + b_channel) / 2) / (r_channel + g_channel + b_channel + 1e-6)
    redness_score = np.mean(redness)
    
    # Color uniformity (dehydrated lips are more patchy)
    color_variance = np.mean([np.var(img_np[:, :, i]) for i in range(3)])
    
    return {
        'rgb_mean_r': float(rgb_mean[0]),
        'rgb_mean_g': float(rgb_mean[1]),
        'rgb_mean_b': float(rgb_mean[2]),
        'rgb_std': float(np.mean(rgb_std)),
        'hsv_hue': float(hsv_mean[0]),
        'hsv_saturation': float(hsv_mean[1]),
        'hsv_value': float(hsv_mean[2]),
        'lab_lightness': float(lab_mean[0]),
        'lab_a': float(lab_mean[1]),
        'lab_b': float(lab_mean[2]),
        'redness_ratio': float(redness_score),
        'color_uniformity': float(1.0 / (color_variance + 1))  # Higher is more uniform
    }


# ======================================================
# 3. TEXTURE ANALYSIS (CRACK & DRYNESS DETECTION)
# ======================================================
def analyze_lip_texture(image):
    """
    Detect cracks, roughness, and dry patches.
    Returns dict of texture features.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # 1. Edge Detection (cracks show as strong edges)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 2. Texture Variance using Laplacian (smooth vs rough)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = np.var(laplacian)
    laplacian_mean = np.abs(np.mean(laplacian))
    
    # 3. Local Binary Pattern (texture descriptor)
    # Captures micro-texture patterns
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    
    # 4. Gradient Magnitude (texture strength)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_mean = np.mean(gradient_magnitude)
    
    # 5. Standard Deviation (local variance indicates roughness)
    local_std = cv2.blur(gray, (5, 5))
    texture_roughness = np.std(local_std)
    
    # Crack severity score (0-100)
    # High edge density + high gradient = more cracks
    crack_score = min(100, (edge_density * 1000 + gradient_mean / 10))
    
    return {
        'crack_density': float(edge_density),
        'crack_severity_score': float(crack_score),
        'surface_roughness': float(laplacian_var),
        'texture_strength': float(gradient_mean),
        'texture_variation': float(texture_roughness),
        'lbp_entropy': float(-np.sum(lbp_hist * np.log(lbp_hist + 1e-10))),  # Texture complexity
        'edge_sharpness': float(laplacian_mean)
    }


# ======================================================
# 4. AUTOMATIC IMAGE ENHANCEMENT
# ======================================================
def auto_adjust_image(image):
    """
    Enhance lighting and contrast automatically.
    Returns enhanced PIL Image.
    """
    img_np = np.array(image)
    
    # Convert to HSV for better control
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    # 1. Brightness adjustment using histogram equalization
    # Only adjust V channel to preserve color
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    
    # 2. Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # 3. Denoise slightly to reduce sensor noise
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 6, 6, 7, 21)
    
    return Image.fromarray(enhanced)


# ======================================================
# 5. COMBINED FEATURE EXTRACTION
# ======================================================
def extract_all_features(image, auto_enhance=True):
    """
    Main feature extraction pipeline.
    Returns dict with all features + metadata.
    """
    # Step 1: Auto-enhance if needed
    original_image = image
    if auto_enhance:
        image = auto_adjust_image(image)
    
    # Step 2: Detect and crop lips
    cropped_image, lip_detected, landmarks = detect_and_crop_lips(image)
    
    # Step 3: Extract features from cropped region
    color_features = extract_color_features(cropped_image)
    texture_features = analyze_lip_texture(cropped_image)
    
    # Combine all features
    all_features = {
        **color_features,
        **texture_features,
        'lip_detected': lip_detected,
        'image_enhanced': auto_enhance,
        'crop_success': lip_detected
    }
    
    return {
        'features': all_features,
        'processed_image': cropped_image,
        'enhanced_image': image if auto_enhance else original_image,
        'landmarks': landmarks,
        'metadata': {
            'original_size': original_image.size,
            'processed_size': cropped_image.size,
            'lip_region_detected': lip_detected
        }
    }


# ======================================================
# 6. FEATURE-BASED QUALITY SCORE
# ======================================================
def calculate_image_quality_score(features):
    """
    Calculate overall image quality score (0-100).
    Used for pre-filtering bad images.
    """
    score = 100.0
    
    # Penalize if lip not detected
    if not features.get('lip_detected', False):
        score -= 30
    
    # Check brightness (LAB L channel)
    lightness = features.get('lab_lightness', 128)
    if lightness < 50:
        score -= 20  # Too dark
    elif lightness > 230:
        score -= 15  # Too bright
    
    # Check color uniformity
    uniformity = features.get('color_uniformity', 0.5)
    if uniformity < 0.3:
        score -= 10  # Too patchy/blurry
    
    # Check if image is too blurry (low texture)
    texture_strength = features.get('texture_strength', 10)
    if texture_strength < 5:
        score -= 15  # Too blurry
    
    return max(0, min(100, score))
