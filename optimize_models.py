
import torch
import torch.nn as nn
from torchvision import models
import os
import sys

# Add paths for modules
sys.path.append(os.path.join(os.getcwd(), 'Mental-H', 'Emotional'))
sys.path.append(os.getcwd())

def optimize_lip_model():
    print("Optimizing Lip Analysis Model (MobileNetV2)...")
    try:
        import hydration.imagePredict_mobilenet as ipm
        
        # Define model architectures
        ipm.define_models()
        
        model_path = os.path.join("hydration", "models", "LipModel_MobileNetV2.pth")
        
        if not os.path.exists(model_path):
             # Try absolute path from cwd
             candidate = os.path.join(os.getcwd(), model_path)
             if os.path.exists(candidate):
                 model_path = candidate
             else:
                 print(f"Skipping: {model_path} not found.")
                 return

        print(f"Loading {model_path}...")
        state_dict = torch.load(model_path, map_location='cpu')
        
        model = None
        
        # Try finding the right architecture
        # 1. SimpleLipModel
        try:
            print("Trying SimpleLipModel...")
            m = ipm.SimpleLipModel(num_classes=2)
            m.load_state_dict(state_dict)
            model = m
            print("Detected: SimpleLipModel")
        except Exception:
            pass

        # 2. ImprovedLipModel
        if model is None:
            try:
                print("Trying ImprovedLipModel...")
                m = ipm.ImprovedLipModel(num_classes=2)
                m.load_state_dict(state_dict)
                model = m
                print("Detected: ImprovedLipModel")
            except Exception:
                pass

        # 3. ExpertLipModel
        if model is None:
            try:
                print("Trying ExpertLipModel...")
                m = ipm.ExpertLipModel(num_classes=2)
                m.load_state_dict(state_dict)
                model = m
                print("Detected: ExpertLipModel")
            except Exception:
                pass

        if model is None:
            print("Failed to load state dict into any known architecture.")
            return

        model.eval()
        
        # Apply Dynamic Quantization
        print("Quantizing model (Float32 -> Int8)...")
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        
        # Save
        save_path = model_path.replace(".pth", "_quantized.pth")
        torch.save(quantized_model.state_dict(), save_path)
        
        # Compare sizes
        orig_size = os.path.getsize(model_path) / (1024 * 1024)
        new_size = os.path.getsize(save_path) / (1024 * 1024)
        print(f"Success! Reduced size from {orig_size:.2f} MB to {new_size:.2f} MB")
        
    except Exception as e:
        print(f"Error optimizing lip model: {e}")
        import traceback
        traceback.print_exc()

def optimize_emotion_model():
    print("\nOptimizing Facial Emotion Model (EmotionCNN)...")
    try:
        from other import EmotionCNN
        
        model_path = os.path.join("Mental-H", "Emotional", "emotion_model.pth")
        
        if not os.path.exists(model_path):
             candidate = os.path.join(os.getcwd(), model_path)
             if os.path.exists(candidate):
                 model_path = candidate
             else:
                 print(f"Skipping: {model_path} not found.")
                 return

        print(f"Loading {model_path}...")
        model = EmotionCNN()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Quantize
        print("Quantizing model...")
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        
        save_path = model_path.replace(".pth", "_quantized.pth")
        torch.save(quantized_model.state_dict(), save_path)
        
        orig_size = os.path.getsize(model_path) / (1024 * 1024)
        new_size = os.path.getsize(save_path) / (1024 * 1024)
        print(f"Success! Reduced size from {orig_size:.2f} MB to {new_size:.2f} MB")

    except Exception as e:
        print(f"Error optimizing emotion model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    optimize_lip_model()
    optimize_emotion_model()
