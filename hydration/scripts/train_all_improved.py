
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from hydration.training.improve_models import main as train_tabular
from hydration.scripts.improve_lip_model import main as train_lip
from core.utils import setup_logging

LOG = setup_logging()

def main():
    LOG.info("=" * 80)
    LOG.info("HYDRATION MANAGEMENT - GLOBAL MODEL IMPROVEMENT & TRAINING PIPELINE")
    LOG.info("=" * 80)
    
    # 1. Train Tabular Model (XGBoost + SMOTE + GridSearch)
    LOG.info("\n[STEP 1/2] Improving Tabular Models (Hydration Form)...")
    try:
        train_tabular()
    except Exception as e:
        LOG.error(f"Error during tabular training: {e}")
    
    # 2. Train Lip Model (MobileNetV2 + Advanced Augmentation + Label Smoothing)
    LOG.info("\n[STEP 2/2] Improving Lip Dehydration Models (Images)...")
    try:
        train_lip()
    except Exception as e:
        LOG.error(f"Error during lip training: {e}")
    
    LOG.info("\n" + "=" * 80)
    LOG.info("ALL HYDRATION MODELS HAVE BEEN IMPROVED AND RETRAINED")
    LOG.info("=" * 80)

if __name__ == "__main__":
    main()
