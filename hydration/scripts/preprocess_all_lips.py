import os
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import numpy as np
from tqdm import tqdm
from hydration.mediapipe_utils import LipExtractor

def preprocess_dataset(input_dir, output_dir):
    extractor = LipExtractor()
    classes = ["Dehydrate", "Normal"]
    
    for cls in classes:
        in_path = os.path.join(input_dir, cls)
        out_path = os.path.join(output_dir, cls)
        os.makedirs(out_path, exist_ok=True)
        
        print(f"Processing class: {cls}")
        files = [f for f in os.listdir(in_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        success_count = 0
        fail_count = 0
        
        for filename in tqdm(files):
            img_path = os.path.join(in_path, filename)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Failed to read {img_path}")
                fail_count += 1
                continue
                
            coords, mask, roi = extractor.extract_lips(image)
            
            if roi is not None:
                # Resize ROI to a standard size for training (optional, but good for consistency)
                # The training script resizing will handle this, but saving it as is works too.
                # However, MediaPipe sometimes returns very small crops. 
                # Let's save the ROI.
                save_path = os.path.join(out_path, filename)
                cv2.imwrite(save_path, roi)
                success_count += 1
            else:
                # print(f"No lips detected in {img_path}")
                fail_count += 1
        
        print(f"Class {cls} Summary: Success={success_count}, Failed={fail_count}")
    
    print("\n✅ Preprocessing complete!")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(PARENT_DIR, "data")
    PROCESSED_DATA_DIR = os.path.join(PARENT_DIR, "data_processed")
    
    preprocess_dataset(DATA_DIR, PROCESSED_DATA_DIR)
