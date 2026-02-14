import cv2
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

# ======================================================
# NORMALIZATION
# ======================================================
def normalize_exercise_name(name):
    if not name:
        return "unknown"
    return name.lower().strip().replace(" ", "_").replace("-", "_")

# ======================================================
# UI
# ======================================================
def create_compact_info_panel(width, exercise, form, confidence, stage, colors,
                               reps=0, hold_time=0, show_heatmap=True, frame_count=0,
                               last_rep_status="unknown", reps_correct=0, reps_wrong=0):

    panel = np.zeros((90, width, 3), dtype=np.uint8)

    exercise = normalize_exercise_name(exercise)

    color = colors.get(form, colors["unknown"])
    form_text = form.upper()

    cv2.putText(panel, f"EXERCISE: {exercise.upper()}",
                (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(panel, f"FORM: {form_text}",
                (10, 55), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

    if exercise == "plank":
        rep_text = f"HOLD: {hold_time}s"
        # No correct/wrong for plank yet (unless form check implies hold quality)
        cv2.putText(panel, rep_text,
                    (width // 2 - 70, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        # Draw status box for last rep
        box_color = (50, 50, 50)
        if last_rep_status == "correct":
            box_color = (0, 200, 0) # Green
        elif last_rep_status == "wrong":
            box_color = (0, 0, 200) # Red
        
        cv2.rectangle(panel, (width // 2 - 80, 10), (width // 2 + 80, 60), box_color, -1)
        
        rep_text = f"REPS: {reps}"
        cv2.putText(panel, rep_text,
                    (width // 2 - 50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
        # Show stats
        stats_text = f"C: {reps_correct} | W: {reps_wrong}"
        cv2.putText(panel, stats_text,
                    (width // 2 - 60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.putText(panel, f"Confidence: {confidence:.1f}%",
                (width - 260, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    heatmap_status = "ON" if show_heatmap else "OFF"
    cv2.putText(panel, f"Heatmap: {heatmap_status} | Frame: {frame_count}",
                (width - 260, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return panel


def draw_progress_bar(frame, current, total):
    if total <= 0:
        return
    bar_width, bar_height = 400, 15
    bar_x = (frame.shape[1] - bar_width) // 2
    bar_y = frame.shape[0] - 30

    progress = min(max(current / total, 0), 1)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 200, 0), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)

# ======================================================
# FEATURE EXTRACTION (132 FEATURES ONLY)
# ======================================================
def extract_features(landmarks):
    features = np.zeros(132, dtype=np.float32)

    if not landmarks or not hasattr(landmarks, "landmark"):
        return features.tolist()

    for i, lm in enumerate(landmarks.landmark[:33]):
        idx = i * 4
        features[idx:idx+4] = [lm.x, lm.y, lm.z, lm.visibility]

    return features.tolist()

# 🔥 FINAL FIX: STRICT FEATURE MATCH
def create_feature_dataframe(features, expected_features):
    # DETECT NAMING CONVENTION based on first feature
    # Expected features e.g. ['x_0', 'y_0'...] OR ['x0', 'y0'...]
    use_underscore = True
    if expected_features and len(expected_features) > 0:
        if "x0" in expected_features:
            use_underscore = False
        # Default to True (x_0) as that is what our model expects
    
    data = {}
    for i in range(33):
        base = i * 4
        if use_underscore:
            data[f"x_{i}"] = features[base]
            data[f"y_{i}"] = features[base + 1]
            data[f"z_{i}"] = features[base + 2]
            data[f"v_{i}"] = features[base + 3]
        else:
            data[f"x{i}"] = features[base]
            data[f"y{i}"] = features[base + 1]
            data[f"z{i}"] = features[base + 2]
            data[f"v{i}"] = features[base + 3]

    X = pd.DataFrame([data])

    # EXACT MATCH WITH TRAINING FEATURES
    # Ensure we include all columns the model expects, filling missing with 0
    X = X.reindex(columns=expected_features, fill_value=0.0)

    return X

# ======================================================
# FILE HELPERS
# ======================================================
def find_video_files():
    exts = (".mp4", ".avi", ".mov", ".mkv")
    files = [f for f in os.listdir(".") if f.lower().endswith(exts)]
    if os.path.exists("Videos"):
        files += [f"Videos/{f}" for f in os.listdir("Videos") if f.lower().endswith(exts)]
    return files

def load_recommendations(json_file="recommendations.json"):
    try:
        with open(json_file, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_detailed_report_with_recommendations(*args, **kwargs):
    return
