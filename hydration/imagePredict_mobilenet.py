# ======================================================
# DEPENDENCY HANDLING (Lazy Loading for Memory Optimization)
# ======================================================
from core.config import DEVICE, MOBILENET_MODEL_OUT

# Constants
ADVANCED_FEATURES_AVAILABLE = False # Will check inside functions


# ======================================================
# QUALITY CHECKS
# ======================================================
def check_image_quality(image):
    """
    Checks if image is too dark or has low variance (blur/flat).
    Returns (Passed: bool, Reason: str)
    
    🔥 VERY LENIENT: Only reject extremely poor quality images
    """
    from PIL import ImageStat
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    
    # 1. Brightness Check (Very lenient - only reject extreme cases)
    brightness = stat.mean[0]
    if brightness < 15:  # Extremely dark (was 25)
        return False, f"Image too dark (Brightness: {brightness:.1f}/255)"
    elif brightness > 250:  # Overexposed
        return False, f"Image too bright/overexposed (Brightness: {brightness:.1f}/255)"
    
    # 2. Blur/Contrast Check (Very lenient - only reject extreme cases)
    variance = stat.var[0]
    if variance < 20:   # Extremely low detail (was 50)
        return False, f"Image likely blurry or low contrast (Variance: {variance:.1f})"
    
    # Log warnings for borderline cases but don't reject
    if brightness < 40:
        print(f"[WARN] Image is dark (Brightness: {brightness:.1f}) but accepting")
    if variance < 100:
        print(f"[WARN] Image may be blurry (Variance: {variance:.1f}) but accepting")
    
    # If we get here, image passes quality checks
    return True, "OK"


# ======================================================
# CONTENT RELEVANCE CHECK (SKIN TONE FILTER)
# ======================================================
def check_content_relevance(image):
    """
    Uses HSV color space to check if image contains sufficient skin-tone pixels.
    Returns (Passed: bool, Reason: str)
    
    🔥 DISABLED: Too many false rejections. Let the ML model decide.
    """
    # DISABLED: This check was rejecting too many valid lip images
    # The ML model is trained to handle various backgrounds and lighting
    # Let the model itself determine if the image is relevant
    
    print(f"[INFO] Content check: SKIPPED (letting ML model decide)")
    return True, "Content check disabled - ML model will validate"
    
    # OLD CODE (commented out):
    # try:
    #     from matplotlib.colors import rgb_to_hsv
    #     img_np = np.array(image)
    #     img_hsv = rgb_to_hsv(img_np / 255.0)
    #     
    #     h = img_hsv[:,:,0]
    #     s = img_hsv[:,:,1]
    #     v = img_hsv[:,:,2]
    #     
    #     skin_mask = ( 
    #         ((h < 0.18) | (h > 0.93)) &
    #         (s > 0.15) &
    #         (s < 0.85) &
    #         (v > 0.25)
    #     )
    #     
    #     skin_pixels = np.sum(skin_mask)
    #     total_pixels = image.width * image.height
    #     ratio = skin_pixels / total_pixels
    #     
    #     if ratio < 0.10:
    #         return False, f"No human skin/lips detected (Skin Ratio: {ratio:.1%})"
    #     
    #     return True, "OK"
    # except Exception as e:
    #     return True, "Check Skipped"


# ======================================================
# LOAD TRAINED MOBILENETV2 MODEL (SUPPORT BOTH ARCHITECTURES)
# ======================================================
# Model class placeholders (defined globally inside define_models() to avoid early torch dependency)
SimpleLipModel = None
ImprovedLipModel = None
ExpertLipModel = None

try:
    from hydration.lip_feature_extractor import extract_all_features, calculate_image_quality_score
    ADVANCED_FEATURES_AVAILABLE = True
except (ImportError, Exception):
    ADVANCED_FEATURES_AVAILABLE = False

def define_models():
    """Defines model classes inside a function to avoid global torch dependency"""
    global SimpleLipModel, ImprovedLipModel, ExpertLipModel
    import torch.nn as nn
    from torchvision import models

    class SimpleLipModel(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.mobilenet = models.mobilenet_v2(pretrained=False)
            num_ftrs = self.mobilenet.classifier[1].in_features
            self.mobilenet.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_ftrs, num_classes)
            )
        def forward(self, x):
            return self.mobilenet(x)

    class ImprovedLipModel(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.mobilenet = models.mobilenet_v2(pretrained=False)
            num_ftrs = self.mobilenet.classifier[1].in_features
            self.mobilenet.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        def forward(self, x):
            return self.mobilenet(x)

    class ExpertLipModel(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.mobilenet = models.mobilenet_v2(pretrained=False)
            num_ftrs = self.mobilenet.classifier[1].in_features
            self.mobilenet.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
        def forward(self, x):
            return self.mobilenet(x)

def load_model(class_names):
    import os
    import torch
    import gc
    from PIL import Image
    
    # Ensure models are defined
    define_models()
    """
    Load the trained MobileNetV2 model for lip hydration prediction.
    """
    print(f"DEBUG: Starting Load from {MOBILENET_MODEL_OUT}")
    if not os.path.exists(MOBILENET_MODEL_OUT):
        raise FileNotFoundError(f"CRITICAL: Model file not found! Expected: {MOBILENET_MODEL_OUT}")

    num_classes = len(class_names)
    gc.collect() # Pre-load cleanup

    try:
        # Load state_dict to CPU first
        state_dict = torch.load(MOBILENET_MODEL_OUT, map_location="cpu", weights_only=True)
        
        # Checkpoint prefix fix
        if state_dict and not any(k.startswith("mobilenet.") for k in state_dict.keys()):
            state_dict = {f"mobilenet.{k}": v for k, v in state_dict.items()}

        # Architecture detection
        weight_1 = state_dict.get("mobilenet.classifier.1.weight")
        if weight_1 is None: weight_1 = state_dict.get("classifier.1.weight")
        
        weight_9 = state_dict.get("mobilenet.classifier.9.weight")
        if weight_9 is None: weight_9 = state_dict.get("classifier.9.weight")
        
        weight_13 = state_dict.get("mobilenet.classifier.13.weight")
        if weight_13 is None: weight_13 = state_dict.get("classifier.13.weight")

        if weight_1 is not None and weight_1.shape[0] == num_classes and weight_1.shape[1] == 1280:
            model = SimpleLipModel(num_classes=num_classes)
        elif weight_13 is not None and weight_13.shape[0] == num_classes and weight_13.shape[1] == 128:
            model = ExpertLipModel(num_classes=num_classes)
        elif weight_9 is not None and weight_9.shape[0] == num_classes and weight_9.shape[1] == 256:
            model = ImprovedLipModel(num_classes=num_classes)
        else:
            print("[WARN] Unknown architecture, using ImprovedLipModel")
            model = ImprovedLipModel(num_classes=num_classes)

        # Load into model
        model.load_state_dict(state_dict, strict=False)
        
        # CRITICAL: Delete state_dict and collect garbage IMMEDIATELY
        del state_dict
        gc.collect()
        
        model.to(DEVICE)
        model.eval()
        print(f"[OK] Model ready on {DEVICE}")
        return model
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Failed to load model: {e}")


# ======================================================
# RECOMMENDATION LOGIC
# ======================================================
def get_recommendation(label_status, confidence):
    if label_status == "Dehydrate":
        return (
            "⚠️ Possible Dehydration Detected.\n"
            f"   (Confidence: {confidence:.0%})\n"
            "- Drink 1–2 glasses of water immediately.\n"
            "- Avoid caffeine/alcohol for 2 hours.\n"
            "- Check if lips feel dry or cracked."
        )
    elif label_status == "Uncertain":
        return (
            "⚠️ Results Inconclusive.\n"
            "- The model is not confident.\n"
            "- Please try again with better lighting."
        )
    elif label_status == "REJECTED":
        return (
            "❌ Prediction Aborted.\n"
            "- The image does not appear to contain a human face/lips.\n"
            "- Please use a clear close-up of the lip area."
        )
    else:
        return (
            "✅ Hydration appears normal.\n"
            "- Keep maintaining regular water intake."
        )


# ======================================================
# HYDRATION SCORE (0–100)
# ======================================================
def calculate_hydration_score(label, confidence):
    """
    Calculate hydration score (0-100) based on prediction.
    
    🔥 IMPROVED: More nuanced scoring that better reflects hydration state
    
    Score Ranges:
    - 80-100: Excellent hydration
    - 60-79: Good hydration
    - 40-59: Mild dehydration
    - 20-39: Moderate dehydration
    - 0-19: Severe dehydration
    """
    if label == "Dehydrate":
        # Lower score = Worse hydration
        # High confidence dehydration = lower score
        # Confidence 1.0 -> score 10 (severe)
        # Confidence 0.5 -> score 35 (mild-moderate)
        base_score = int((1 - confidence) * 50)
        score = max(10, min(45, base_score + 10))  # Clamp to 10-45 range
    else:
        # Higher score = Better hydration
        # High confidence normal = higher score
        # Confidence 1.0 -> score 95 (excellent)
        # Confidence 0.5 -> score 70 (good)
        base_score = int(55 + confidence * 40)
        score = max(60, min(95, base_score))  # Clamp to 60-95 range
    
    return score


# ======================================================
# UI OVERLAY
# ======================================================
def draw_overlay(image, score, status, warnings=[]):
    from PIL import Image, ImageDraw, ImageFont
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Colors
    if status == "Dehydrate":
        bg_col = (200, 50, 50, 180) # Red
    elif status == "Uncertain":
        bg_col = (200, 160, 50, 180) # Orange
    elif status == "REJECTED":
        bg_col = (80, 80, 80, 200) # Gray
    else:
        bg_col = (50, 180, 80, 180) # Green

    # Top Banner
    draw.rectangle((0, 0, image.width, 60), fill=bg_col)
    
    # Try to load font, fallback to default
    try:
        font_lg = ImageFont.truetype("arial.ttf", 26)
        font_sm = ImageFont.truetype("arial.ttf", 16)
    except:
        font_lg = ImageFont.load_default()
        font_sm = ImageFont.load_default()

    draw.text((20, 15), f"Status: {status}", fill="white", font=font_lg)

    # Warnings (if any)
    y_warn = 70
    for w in warnings:
        draw.rectangle((0, y_warn, image.width, y_warn + 30), fill=(0, 0, 0, 150))
        draw.text((20, y_warn + 5), f"⚠️ {w}", fill="yellow", font=font_sm)
        y_warn += 35

    return Image.alpha_composite(image, overlay).convert("RGB")



# ======================================================
# GRAD-CAM VISUALIZATION (XAI)
# ======================================================
def generate_gradcam_heatmap(model, input_tensor, target_class, original_image):
    """
    Manual, lightweight Grad-CAM implementation.
    Optimized for low-RAM environments (Render Free Tier).
    """
    import numpy as np
    import cv2
    import torch
    import gc
    from PIL import Image

    activations = None
    gradients = None

    def save_activations(module, input, output):
        nonlocal activations
        activations = output

    def save_gradients(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    try:
        # Clear any lingering gradients
        model.zero_grad()
        
        # Target the last conv layer of MobileNetV2
        target_layer = model.mobilenet.features[18]
        
        # Register hooks
        f_hook = target_layer.register_forward_hook(save_activations)
        b_hook = target_layer.register_full_backward_hook(save_gradients)

        # Forward pass with gradients enabled
        with torch.enable_grad():
            x = input_tensor.clone().detach().requires_grad_(True)
            output = model(x)
            
            # Ensure target_class is valid
            if target_class >= output.shape[1]:
                target_class = output.argmax(dim=1).item()
                
            score = output[0, target_class]
            score.backward()

        # Remove hooks immediately
        f_hook.remove()
        b_hook.remove()

        if activations is None or gradients is None:
            return None, "Reasoning visualization failed."

        # Compute Grad-CAM
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = torch.relu(cam)
        
        heatmap = cam.cpu().detach().numpy()
        
        # Cleanup Tensors IMMEDIATELY
        del x, output, score, weights, cam, activations, gradients
        gc.collect()

        # Normalize 
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

        # Region explanation
        h, w = heatmap.shape
        gy, gx = h // 3, w // 3
        v_top = np.mean(heatmap[0:gy, :])
        v_bottom = np.mean(heatmap[2*gy:, :])
        v_center = np.mean(heatmap[gy:2*gy, gx:2*gx])
        
        if v_top > v_bottom and v_top > v_center: region = "upper lip"
        elif v_bottom > v_top and v_bottom > v_center: region = "lower lip"
        else: region = "central lip area"
        
        explanation = f"AI focused on the {region} texture to determine hydration status."

        # Resize and Color
        overlay_heatmap = cv2.resize(heatmap, (original_image.width, original_image.height))
        overlay_heatmap = np.uint8(255 * overlay_heatmap)
        heatmap_img = cv2.applyColorMap(overlay_heatmap, cv2.COLORMAP_JET)
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
        
        # Overlay
        original_np = np.array(original_image)
        overlay = cv2.addWeighted(original_np, 0.7, heatmap_img, 0.3, 0)
        
        return Image.fromarray(overlay), explanation
    except Exception as e:
        print(f"[XAI Error] Manual Grad-CAM failed: {e}")
        return None, "Reasoning visualization unavailable."

# ======================================================
# IMAGE PREDICTION (ENHANCED WITH ADVANCED FEATURES)
# ======================================================
def predict_image(image_path, model, class_names):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import os
    from datetime import datetime
    from hydration.training.preprocess_images import get_transforms
    
    transform = get_transforms(train=False)
    
    try:
        # MEMORY OPTIMIZATION: Open and immediately resize to a manageable size
        # This prevents large 10MB+ images from hogging RAM during processing
        with Image.open(image_path) as img_raw:
             # Resize to max 800px on longest side for processing
             MAX_PROCESS_SIZE = 800
             w, h = img_raw.size
             if max(w, h) > MAX_PROCESS_SIZE:
                 scale = MAX_PROCESS_SIZE / max(w, h)
                 img_raw = img_raw.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
             image = img_raw.convert("RGB")
    except Exception as e:
        print(f"[Error] Could not open image: {e}")
        return None

    warnings = []
    advanced_info = {}
    
    # ========== ADVANCED FEATURE EXTRACTION (MEDIAPIPE) ==========
    # Track RAM usage: Mediapipe can cause OOM on Render Free (512MB)
    if ADVANCED_FEATURES_AVAILABLE:
        try:
            print("[INFO] Running advanced feature extraction...")
            feature_data = extract_all_features(image, auto_enhance=True)
            
            features = feature_data['features']
            processed_image = feature_data['processed_image']
            enhanced_image = feature_data['enhanced_image']
            landmarks = feature_data['landmarks']
            metadata = feature_data['metadata']
            
            # Calculate quality score
            quality_score = calculate_image_quality_score(features)
            
            # Store advanced info for output
            advanced_info = {
                'quality_score': quality_score,
                'lip_detected': features.get('lip_detected', False),
                'crack_severity': features.get('crack_severity_score', 0),
                'color_redness': features.get('redness_ratio', 0),
                'texture_roughness': features.get('surface_roughness', 0),
                'landmarks': landmarks
            }
            
            # Quality-based warnings
            if quality_score < 60:
                warnings.append(f"Image Quality: {quality_score:.0f}/100")
            
            if not features.get('lip_detected'):
                warnings.append("Lip region not clearly detected")
            
            # Use processed (cropped + enhanced) image for prediction
            image_for_prediction = processed_image
            
            print(f"[INFO] Quality Score: {quality_score:.1f}/100")
            print(f"[INFO] Lip Detected: {features.get('lip_detected')}")
            print(f"[INFO] Crack Severity: {features.get('crack_severity_score', 0):.1f}")
            
        except Exception as e:
            print(f"[Warning] Advanced features (Mediapipe) failed or OOM prevented: {e}")
            image_for_prediction = image
            warnings.append("Advanced analysis skipped (memory saver)")
    else:
        image_for_prediction = image
    
    # 1. Quality Check
    is_good, reason = check_image_quality(image)
    if not is_good:
        print(f"[Warning] Image Quality Issue: {reason}")
        warnings.append(reason)

    # 2. Content Relevance Check
    is_relevant, relevance_reason = check_content_relevance(image_for_prediction)
    if not is_relevant:
        print(f"[REJECTING] {relevance_reason}")
        # Stop Pipeline Here
        status_display = "REJECTED"
        warnings.append(relevance_reason)
        score = 0
        final_image = draw_overlay(image, score, status_display, warnings)
        
        # Save & Return Early
        os.makedirs("img/uploads", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"img/uploads/rejected_{ts}.png"
        final_image.save(out_path)
        
        return status_display, 0, 0.0, get_recommendation("REJECTED", 0), out_path, None, "Image rejected due to quality issues", advanced_info

    # 3. Inference
    print("[INFO] Running PyTorch Inference...")
    tensor = transform(image_for_prediction).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        
    p_dehydrate = probs[0][0].item()
    p_normal = probs[0][1].item()
    
    # Determine final label based on thresholds
    DEHYDRATION_THRESHOLD = 0.45  # More balanced decision boundary
    UNCERTAINTY_THRESHOLD = 0.55   # Only mark truly ambiguous cases as uncertain
    
    # Adjust prediction based on advanced features (if available)
    if ADVANCED_FEATURES_AVAILABLE and 'crack_severity' in advanced_info:
        crack_severity = advanced_info['crack_severity']
        
        # If high crack severity detected, boost dehydration confidence
        if crack_severity > 30:
            print(f"[INFO] High crack severity ({crack_severity:.1f}) detected - adjusting prediction")
            
            boost_factor = 0.15
            p_dehydrate += boost_factor
            
            # Renormalize probabilities to sum to 1.0
            total_prob = p_dehydrate + p_normal
            if total_prob > 0:
                p_dehydrate = p_dehydrate / total_prob
                p_normal = p_normal / total_prob
            
            print(f"[INFO] Adjusted probabilities - Dehydrate: {p_dehydrate:.2%}, Normal: {p_normal:.2%}")

    if p_dehydrate > DEHYDRATION_THRESHOLD:
        label = "Dehydrate"
        confidence = p_dehydrate
    else:
        label = "Normal"
        confidence = p_normal

    if confidence < UNCERTAINTY_THRESHOLD:
        status_display = "Uncertain"
        warnings.append(f"Low Confidence ({confidence:.0%})")
        print(f"[WARNING] Uncertain prediction - Confidence: {confidence:.0%}")
    else:
        status_display = label
        print(f"[INFO] Confident prediction - {status_display} ({confidence:.0%})")

    score = calculate_hydration_score(label, confidence)
    
    # 4. Grad-CAM Visualization (XAI)
    # Memory Guard: We use explicit garbage collection inside generate_gradcam_heatmap
    heatmap_pil = None
    xai_desc = "Texture and color analysis used for reasoning."
    
    try:
        print("[INFO] Generating XAI Heatmap...")
        heatmap_pil, xai_reasoning = generate_gradcam_heatmap(model, tensor, class_names.index(label), image_for_prediction)
        if xai_reasoning:
             xai_desc = xai_reasoning
             
        # Enhance XAI description with advanced features
        if ADVANCED_FEATURES_AVAILABLE and advanced_info:
            xai_additions = []
            if advanced_info.get('crack_severity', 0) > 20:
                xai_additions.append(f"Surface texture analysis detected signs of dryness.")
            if advanced_info.get('color_redness', 0) > 0.1:
                xai_additions.append("Color analysis shows increased redness.")
            if xai_additions:
                xai_desc = xai_desc + " " + " ".join(xai_additions)
    except Exception as e:
        print(f"[Warning] XAI failed: {e}")
    
    final_image = draw_overlay(image, score, status_display, warnings)
    
    os.makedirs("img/uploads", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    out_path = f"img/uploads/result_{ts}.png"
    final_image.save(out_path)
    
    xai_path = f"img/uploads/xai_heatmap_{ts}.png" if heatmap_pil else None
    if heatmap_pil:
        heatmap_pil.save(xai_path)
    
    # CRITICAL: Clean up tensors to free memory
    del tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return status_display, score, confidence, get_recommendation(status_display, confidence), out_path, xai_path, xai_desc, advanced_info


# ======================================================
# LAZY LOADER GLOBAL
# ======================================================
GLOBAL_MODEL = None
GLOBAL_CLASSES = ["Dehydrate", "Normal"]

def predict_single(image_path):
    global GLOBAL_MODEL
    print(f"DEBUG: Entering predict_single for {image_path}")
    if GLOBAL_MODEL is None:
        try:
            print("DEBUG: Loading GLOBAL_MODEL...")
            GLOBAL_MODEL = load_model(GLOBAL_CLASSES)
        except Exception as e:
            print(f"DEBUG: Model Load Error: {e}")
            return {"error": f"Model load failed: {str(e)}", "confidence": 0, "hydration_score": 0, "prediction": "Error"}

    print("DEBUG: Model loaded, proceeding to predict_image...")
    result = predict_image(image_path, GLOBAL_MODEL, GLOBAL_CLASSES)
    
    # Handle both old and new return formats
    if result is None:
         return {"error": "Prediction pipeline returned None", "prediction": "Error"}
         
    if len(result) == 8:
        label, score, conf, rec, saved_path, xai_path, xai_desc, advanced_info = result
    else:
        label, score, conf, rec, saved_path = result[:5]
        xai_path = result[5] if len(result) > 5 else None
        xai_desc = result[6] if len(result) > 6 else "No description"
        advanced_info = {}
    
    print(f"DEBUG: Prediction successful: {label}")
    return {
        "prediction": label,
        "hydration_score": score,
        "confidence": float(conf),
        "saved_image_path": saved_path,
        "xai_heatmap_path": xai_path,
        "xai_description": xai_desc,
        "recommendation": rec,
        "advanced_analysis": advanced_info
    }
