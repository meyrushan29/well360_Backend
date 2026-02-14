import logging
import sys
import pickle
import joblib  # For loading ML models
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import json
from datetime import datetime

def setup_logging(level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger('hydration_ml')
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler('hydration_ml.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_pickle(obj: Any, path: Path) -> None:

    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise IOError(f"Failed to save pickle to {path}: {e}")


def load_pickle(path: Path) -> Any:
    """
    Load a pickled object using joblib (consistent with training script).
    
    Note: We use joblib.load() instead of pickle.load() because:
    - Training scripts use joblib.dump() to save models
    - joblib is optimized for sklearn/numpy objects
    - Mixing pickle/joblib can cause "invalid load key" errors
    """
    try:
        return joblib.load(path)
    except Exception as e:
        raise IOError(f"Failed to load pickle from {path}: {e}")


def ensure_dir(path: Path) -> Path:

    path.mkdir(parents=True, exist_ok=True)
    return path


def calculate_model_metrics(y_true, y_pred, model_type: str) -> Dict[str, float]:

    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )

    metrics = {}

    if model_type == 'regression':
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    elif model_type == 'classification':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Confusion matrix as dictionary
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

    return metrics


def save_metrics(metrics: Dict, path: Path) -> None:

    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)


class Timer:


    def __enter__(self):
        self.start = datetime.now()
        return self

    def __exit__(self, *args):
        self.end = datetime.now()
        self.duration = self.end - self.start

    def get_duration(self) -> float:
        return self.duration.total_seconds()


def calculate_unified_score(input_type: str, value: Any, confidence: float = 1.0) -> int:
    """
    Unified Hydration Score (0-100)
    0 = Severely Dehydrated
    100 = Perfectly Hydrated
    """
    score = 50

    if input_type == 'form':
        # Value is recommended_water_liters (0.0 to 4.0+)
        # 0.0L needed -> Score 100
        # 4.0L needed -> Score 0
        rec_liter = float(value)
        if rec_liter <= 0: return 100
        
        # Linear decay: 100 - (liters * 25)
        # 0L -> 100
        # 1L -> 75
        # 2L -> 50
        # 3L -> 25
        # 4L -> 0
        score = 100 - (rec_liter * 25)
        
    elif input_type == 'lip':
        # Value is Label (Normal/Dehydrate), Confidence is 0.0-1.0
        label = str(value).lower()
        if "normal" in label:
            # 50 to 100 based on confidence
            # Conf 0.5 -> 75
            # Conf 1.0 -> 100
            score = 50 + (confidence * 50)
        else:
            # 50 to 0 based on confidence (Dehydrated)
            # Conf 0.5 -> 25
            # Conf 1.0 -> 0
            score = 50 - (confidence * 50)

    return int(max(0, min(100, score)))


# Helper: Convert UTC (Naive from DB) to Local System Time
def to_system_local(dt_utc):
    if dt_utc is None: return None
    # Assume stored as Naive UTC. Access timezone from existing import or new one.
    # Since 'from datetime import datetime' is at top, we need 'timezone'.
    from datetime import timezone
    utc_aware = dt_utc.replace(tzinfo=timezone.utc)
    return utc_aware.astimezone() # Converts to System Local Time

def parse_slot_hours(slot_name):
    slot_name = slot_name.lower().strip()
    # "Midnight-4 AM", "4 AM-8 AM", "8 AM-12 PM", "12 PM-4 PM", "4 PM-8 PM", "8 PM-Midnight"
    if "midnight-4 am" in slot_name: return [0, 1, 2, 3]
    if "4 am-8 am" in slot_name: return [4, 5, 6, 7]
    if "8 am-12 pm" in slot_name: return [8, 9, 10, 11]
    if "12 pm-4 pm" in slot_name: return [12, 13, 14, 15]
    if "4 pm-8 pm" in slot_name: return [16, 17, 18, 19]
    if "8 pm-midnight" in slot_name: return [20, 21, 22, 23]
    return []


def fetch_personalized_suggestions(db, model_type: str, prediction_data: Dict) -> list:
    """
    Fetch personalized suggestions from database based on prediction results.
    
    Args:
        db: Database session
        model_type: "form" or "lip"
        prediction_data: Dictionary containing prediction results
        
    Returns:
        List of top 3 matching suggestions sorted by priority (highest first)
        Maximum 3 suggestions for better user experience
    """
    from core.models import HydrationSuggestion
    
    # Start with active suggestions for the correct model type
    query = db.query(HydrationSuggestion).filter(
        HydrationSuggestion.is_active == True,
        (HydrationSuggestion.model_type == model_type) | (HydrationSuggestion.model_type == "both")
    )
    
    all_suggestions = query.all()
    matching_suggestions = []
    
    for suggestion in all_suggestions:
        matches = True
        
        if model_type == "form":
            # Check Form Prediction conditions
            risk_level = prediction_data.get("risk_level")
            recommended_liters = prediction_data.get("recommended_liters")
            activity_level = prediction_data.get("activity_level")
            temperature = prediction_data.get("temperature_c")
            has_any_symptom = prediction_data.get("has_symptoms", False)
            time_slot = prediction_data.get("time_slot")
            
            # Risk level matching
            if suggestion.risk_level and suggestion.risk_level != risk_level:
                matches = False
            
            # Recommended liters range
            if suggestion.min_recommended_liters is not None and recommended_liters < suggestion.min_recommended_liters:
                matches = False
            if suggestion.max_recommended_liters is not None and recommended_liters > suggestion.max_recommended_liters:
                matches = False
            
            # Activity level matching
            if suggestion.activity_level and suggestion.activity_level != activity_level:
                matches = False
            
            # Temperature range
            if suggestion.temperature_min is not None and temperature < suggestion.temperature_min:
                matches = False
            if suggestion.temperature_max is not None and temperature > suggestion.temperature_max:
                matches = False
            
            # Symptom matching
            if suggestion.has_symptoms is not None and suggestion.has_symptoms != has_any_symptom:
                matches = False
            
            # Time slot matching
            if suggestion.time_slots and time_slot not in suggestion.time_slots:
                matches = False
                
        elif model_type == "lip":
            # Check Lip Analysis conditions
            lip_prediction = prediction_data.get("lip_prediction")
            hydration_score = prediction_data.get("hydration_score")
            
            # Lip prediction matching
            if suggestion.lip_prediction and suggestion.lip_prediction != lip_prediction:
                matches = False
            
            # Hydration score range
            if suggestion.min_hydration_score is not None and hydration_score < suggestion.min_hydration_score:
                matches = False
            if suggestion.max_hydration_score is not None and hydration_score > suggestion.max_hydration_score:
                matches = False
        
        if matches:
            matching_suggestions.append({
                "id": suggestion.id,
                "title": suggestion.title,
                "content": suggestion.content,
                "category": suggestion.category,
                "priority": suggestion.priority
            })
    
    # Sort by priority (highest first), then by id
    matching_suggestions.sort(key=lambda x: (-x["priority"], x["id"]))
    
    # 🔥 LIMIT TO TOP 3 SUGGESTIONS FOR BETTER USER EXPERIENCE
    return matching_suggestions[:3]

