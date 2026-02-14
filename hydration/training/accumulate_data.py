
import os
import sys
sys.path.append(os.getcwd()) # Fix module path

import shutil
from sqlalchemy.orm import Session
from core.database import SessionLocal
from core.models import LipAnalysis
from core.config import DATA_DIR

def accumulate_data(confidence_threshold=0.90):
    """
    Harvests images from LipAnalysis table with high specific confidence
    and copies them to the training dataset.
    """
    print(f"--- Data Accumulation (Threshold: {confidence_threshold}) ---")
    
    db: Session = SessionLocal()
    
    try:
        # filter for high confidence
        entries = db.query(LipAnalysis).filter(LipAnalysis.confidence >= confidence_threshold).all()
        
        counts = {"Dehydrate": 0, "Normal": 0}
        
        for entry in entries:
            if not entry.image_path or not os.path.exists(entry.image_path):
                continue
            
            label = entry.prediction
            if label not in counts:
                continue # Unknown label
                
            # Define destination
            dest_dir = os.path.join(DATA_DIR, label)
            os.makedirs(dest_dir, exist_ok=True)
            
            filename = os.path.basename(entry.image_path)
            dest_path = os.path.join(dest_dir, f"harvested_{entry.id}_{filename}")
            
            if not os.path.exists(dest_path):
                shutil.copy2(entry.image_path, dest_path)
                counts[label] += 1
                
        print(f"Harvested Results:")
        print(f"  Normal:    {counts['Normal']} new images")
        print(f"  Dehydrate: {counts['Dehydrate']} new images")
        print(f"Total: {sum(counts.values())} images added to {DATA_DIR}")
        
    except Exception as e:
        print(f"Error extracting data: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    accumulate_data()
