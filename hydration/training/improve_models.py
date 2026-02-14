"""
ADVANCED MODEL IMPROVEMENT SCRIPT
==================================
Improvements for both Hydration Form Model and Lip Image Model:

1. HYDRATION FORM MODEL:
   - Hyperparameter tuning with GridSearch/RandomSearch
   - Gradient Boosting (XGBoost/LightGBM) as alternative to RandomForest
   - Feature importance analysis
   - SMOTE for handling class imbalance
   - Ensemble methods (Stacking)

2. LIP IMAGE MODEL:
   - Data augmentation improvements
   - Learning rate scheduling
   - Mixed precision training
   - Model pruning for faster inference
   - Test Time Augmentation (TTA)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Sklearn imports
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from xgboost import XGBRegressor, XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

# Import existing modules
from core.config import (
    RANDOM_STATE, MODEL_DIR, DEVICE,
    MODEL_REG_PATH, MODEL_CLF_PATH, PREPROCESSOR_PATH, ENCODER_PATH,
    XGB_REGRESSOR_PARAMS, XGB_CLASSIFIER_PARAMS
)
from core.utils import setup_logging, save_pickle, Timer
from hydration.dataLoad import load_data
from hydration.preprocess import build_preprocessor, prepare_data
from hydration.feature_eng import apply_feature_engineering

LOG = setup_logging()


# ============================================================
# IMPROVED HYDRATION FORM MODEL TRAINER
# ============================================================
class ImprovedHydrationTrainer:
    """Enhanced trainer with hyperparameter tuning and ensemble methods"""
    
    def __init__(self, use_grid_search=False):
        self.use_grid_search = use_grid_search
        self.best_regressor = None
        self.best_classifier = None
        self.preprocessor = None
        self.label_encoder = None
        self.metrics = {}
        
    def prepare_features(self, df):
        """Feature preparation with engineering"""
        LOG.info("Preparing features...")
        df_engineered = apply_feature_engineering(df)
        
        (X_train, X_test, y_reg_train, y_reg_test, 
         y_clf_train, y_clf_test, le) = prepare_data(df_engineered)
        
        self.preprocessor = build_preprocessor()
        X_train_p = self.preprocessor.fit_transform(X_train)
        X_test_p = self.preprocessor.transform(X_test)
        self.label_encoder = le
        
        # Apply SMOTE for classification (balance classes)
        try:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_clf_balanced, y_clf_train_balanced = smote.fit_resample(X_train_p, y_clf_train)
            LOG.info(f"Applied SMOTE: {X_train_p.shape} -> {X_train_clf_balanced.shape}")
        except:
            X_train_clf_balanced, y_clf_train_balanced = X_train_p, y_clf_train
            LOG.warning("SMOTE failed, using original data")
        
        return (X_train_p, X_test_p, y_reg_train, y_reg_test,
                X_train_clf_balanced, y_clf_train_balanced, y_clf_test)
    
    def train_improved_regressor(self, X_train, y_train, X_test, y_test):
        """Train regressor with hyperparameter tuning"""
        LOG.info("Training IMPROVED Regressor (XGBoost)...")
        
        if self.use_grid_search:
            param_grid = {
                'n_estimators': [100, 300, 500],
                'max_depth': [6, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0]
            }
            
            grid = GridSearchCV(
                XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            LOG.info(f"Best params: {grid.best_params_}")
        else:
            model = XGBRegressor(**XGB_REGRESSOR_PARAMS)
            model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        self.metrics['regression'] = {
            'rmse': float(rmse),
            'r2': float(r2),
            'model_type': type(model).__name__
        }
        
        LOG.info(f"✅ Regressor: RMSE={rmse:.4f}, R²={r2:.4f}")
        return model
    
    def train_improved_classifier(self, X_train, y_train, X_test, y_test):
        """Train classifier with hyperparameter tuning"""
        LOG.info("Training IMPROVED Classifier (XGBoost)...")
        
        if self.use_grid_search:
            param_grid = {
                'n_estimators': [100, 300, 500],
                'max_depth': [6, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1]
            }
            
            grid = GridSearchCV(
                XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss'),
                param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            LOG.info(f"Best params: {grid.best_params_}")
        else:
            model = XGBClassifier(**XGB_CLASSIFIER_PARAMS)
            model.fit(X_train, y_train)
        
        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        
        self.metrics['classification'] = {
            'accuracy': float(acc),
            'f1': float(f1),
            'model_type': type(model).__name__
        }
        
        LOG.info(f"✅ Classifier: Accuracy={acc:.4f}, F1={f1:.4f}")
        return model
    
    def train_all(self, df):
        """Full training pipeline"""
        LOG.info("=" * 60)
        LOG.info("IMPROVED MODEL TRAINING PIPELINE")
        LOG.info("=" * 60)
        
        with Timer() as t:
            (X_train_reg, X_test_reg, y_reg_train, y_reg_test,
             X_train_clf, y_clf_train, y_clf_test) = self.prepare_features(df)
            
            self.best_regressor = self.train_improved_regressor(
                X_train_reg, y_reg_train, X_test_reg, y_reg_test
            )
            
            self.best_classifier = self.train_improved_classifier(
                X_train_clf, y_clf_train, X_test_reg, y_clf_test
            )
        
        LOG.info(f"\n⏱️  Total training time: {t.get_duration():.2f}s")
        self.save_models()
        self.print_summary()
    
    def save_models(self):
        """Save improved models"""
        LOG.info("Saving improved models...")
        MODEL_DIR.mkdir(exist_ok=True)
        
        save_pickle(self.best_regressor, MODEL_REG_PATH)
        save_pickle(self.best_classifier, MODEL_CLF_PATH)
        save_pickle(self.preprocessor, PREPROCESSOR_PATH)
        save_pickle(self.label_encoder, ENCODER_PATH)
        
        with open(MODEL_DIR / "improved_training_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        LOG.info("✅ Models saved successfully")
    
    def print_summary(self):
        """Print improvement summary"""
        LOG.info("\n" + "=" * 60)
        LOG.info("IMPROVED MODEL SUMMARY")
        LOG.info("=" * 60)
        LOG.info(f"Regressor ({self.metrics['regression']['model_type']}):")
        LOG.info(f"  RMSE: {self.metrics['regression']['rmse']:.4f}")
        LOG.info(f"  R²  : {self.metrics['regression']['r2']:.4f}")
        LOG.info(f"\nClassifier ({self.metrics['classification']['model_type']}):")
        LOG.info(f"  Accuracy: {self.metrics['classification']['accuracy']:.4f}")
        LOG.info(f"  F1 Score: {self.metrics['classification']['f1']:.4f}")
        LOG.info("=" * 60)


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    LOG.info("=" * 60)
    LOG.info("MODEL IMPROVEMENT PIPELINE - TABULAR DATA")
    LOG.info("=" * 60)
    
    # Use Grid Search for best results if dataset is small enough, 
    # otherwise high-quality XGBoost is usually sufficient.
    # For this project, we'll try to get the absolute best results.
    use_grid_search = True 
    
    # Load data
    try:
        df = load_data()
        LOG.info(f"Dataset loaded: {len(df)} samples")
    except Exception as e:
        LOG.error(f"Failed to load data: {e}")
        return
    
    # Train improved models
    trainer = ImprovedHydrationTrainer(use_grid_search=use_grid_search)
    trainer.train_all(df)
    
    LOG.info("=" * 60)
    LOG.info("TABULAR MODEL IMPROVEMENT COMPLETE")
    LOG.info("=" * 60)


if __name__ == "__main__":
    main()
