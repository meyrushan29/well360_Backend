import cv2
import joblib
import mediapipe as mp
import numpy as np
import os
import time
import uuid
import xgboost as xgb
from collections import Counter, deque
from pathlib import Path

# Fix paths to be relative to THIS file
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "Models"

# Add parent directory to path to allow imports from fitness/
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    from .exercise_config import EXERCISE_CONFIG, COLORS
    from .heatmap import HeatmapVisualizer
    from .geometry import calculate_angles_from_landmarks
    from .prediction_config import PredictionConfig, QualityMetrics
    from .utils import (
        create_compact_info_panel,
        draw_progress_bar,
        extract_features,
        create_feature_dataframe,
        load_recommendations,
        save_detailed_report_with_recommendations
    )
except ImportError:
    # Fallback for direct execution/testing
    from exercise_config import EXERCISE_CONFIG, COLORS
    from heatmap import HeatmapVisualizer
    from geometry import calculate_angles_from_landmarks
    from prediction_config import PredictionConfig, QualityMetrics
    from utils import (
        create_compact_info_panel,
        draw_progress_bar,
        extract_features,
        create_feature_dataframe,
        load_recommendations,
        save_detailed_report_with_recommendations
    )

mp_pose = None
mp_draw = None
try:
    # MediaPipe's classic API is `mediapipe.solutions.*`.
    # Some environments accidentally install a different/dist-only `mediapipe` package
    # (or a partial install) that lacks `solutions`. In that case we disable fitness
    # processing rather than crashing the entire backend at import-time.
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils
    else:
        raise ImportError(
            "MediaPipe 'solutions' API not available (missing mediapipe.solutions.pose)."
        )
except Exception as e:
    print(f"WARNING: {e} Fitness features disabled.")
    mp_pose = None
    mp_draw = None

class FitnessVideoProcessor:
    def __init__(self):
        print(f"--> Loading Fitness Models from {MODELS_DIR}")

        if mp_pose is None:
            raise RuntimeError(
                "Fitness processor cannot start: MediaPipe Pose is unavailable. "
                "Reinstall the official 'mediapipe' package compatible with your Python version."
            )
        
        # Load models using absolute paths
        self.model = joblib.load(MODELS_DIR / "exercise_form_detector.pkl")
        
        # FIX: Handle XGBoost version incompatibility (older pickle vs newer lib)
        if hasattr(self.model, "callbacks") is False:
            self.model.callbacks = []
        if hasattr(self.model, "early_stopping_rounds") is False:
            self.model.early_stopping_rounds = None

        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        self.label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
        self.expected_features = joblib.load(MODELS_DIR / "training_features.pkl")
        print(f"DEBUG: Loaded {len(self.expected_features)} expected features.")
        print(f"DEBUG: First 5 expected features: {self.expected_features[:5]}")

        self.recommendations_file = BASE_DIR / "recommendations.json"

        if mp_pose:
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.pose = None
            print("DEBUG: MediaPipe Pose failed to initialize.")

        self.heatmap_viz = HeatmapVisualizer()
        self.reset_state()

    def reset_state(self):
        # Use improved configuration settings
        self.prediction_history = deque(maxlen=PredictionConfig.HISTORY_WINDOW)
        self.confidence_history = deque(maxlen=PredictionConfig.HISTORY_WINDOW)
        self.reps = 0
        self.stage = "up"
        self.last_rep_time = 0
        self.MIN_REP_TIME = 0.8
        self.hold_time = 0
        self.hold_start = None
        self.current_exercise = "unknown"
        self.current_form = "unknown"
        # FIX: Confidence-weighted exercise scores (not just raw counts)
        self.exercise_weighted_scores = {}
        self.exercise_counts = {}
        # FIX: Track form PER exercise (not globally)
        self.exercise_form_counts = {}  # {exercise: {"correct": N, "wrong": M}}
        self.form_counts = {"correct": 0, "wrong": 0, "unknown": 0}  # kept for legacy
        self.predictions = []
        self.no_pose_frames = 0
        self.frame_count = 0
        # FIX: Exercise stability lock - prevents flip-flopping
        self.locked_exercise = None
        self.lock_count = 0
        self.LOCK_THRESHOLD = PredictionConfig.LOCK_THRESHOLD
        # FIX: Minimum confidence to accept a prediction
        self.MIN_CONFIDENCE = PredictionConfig.MIN_CONFIDENCE
        # FIX: Skip early frames (warm-up period)
        self.WARMUP_FRAMES = PredictionConfig.WARMUP_FRAMES
        
        # FIX: Rep counting based on video time
        self.curr_fps = 30 # Default
        self.hold_start_frame = None
        self.last_rep_frame = 0
        self.min_rep_frames = 0 # Will be calculated based on fps
        
        # FIX: Rep Classification
        self.reps_correct = 0
        self.reps_wrong = 0
        self.current_rep_forms = [] # Store forms during a rep
        self.last_rep_status = "unknown" # "correct", "wrong"

        # ======================================================
        # NEW: Advanced Analytics (Separate from ML Model)
        # ======================================================
        self.rep_durations = [] # Time per rep in seconds
        self.rep_roms = []      # Range of Motion % per rep
        self.stability_scores = [] # Variance of hip x-coord
        
        self.current_rep_start_frame = 0
        self.current_rep_min_angle = 180 # Track max depth (min angle)
        self.current_rep_max_angle = 0   # Track max extension
        
        # Stability tracking
        self.hip_x_history = deque(maxlen=30) # 1 sec window at 30fps

    def update_reps(self, landmarks, exercise, current_form="unknown"):
        # ... (unchanged)
        if exercise not in EXERCISE_CONFIG:
            return

        cfg = EXERCISE_CONFIG[exercise]

        if cfg["type"] == "hold":
            if self.hold_start_frame is None:
                self.hold_start_frame = self.frame_count
            
            # Calculate duration based on FRAMES, not wall clock time
            # duration = (current_frame - start_frame) / fps
            frames_elapsed = self.frame_count - self.hold_start_frame
            self.hold_time = int(frames_elapsed / self.curr_fps) if self.curr_fps > 0 else 0
            return

        angles = calculate_angles_from_landmarks(landmarks, cfg)
        if "main" not in angles:
            return

        angle = angles["main"]
        
        # Collect form validity during the "active" part of the rep (stage == "down" usually means flexed/active)
        # Note: "down" in config usually means extended (start), "up" means flexed (end).
        # But logic below says: angle > down -> Extended/Standing. angle < up -> Flexed/Squatting.
        # Usually stage transitions: UP -> DOWN -> UP.
        # We want to track form while they are performing the rep.
        
        if self.stage == "down":
             self.current_rep_forms.append(current_form)

        # LOGIC: Start UP -> go DOWN -> return UP (Count)
        if angle > cfg["down"]: # Extended / Standing (End of rep)
            if self.stage == "down":
                 # Check time based on frames
                 frames_since_last = self.frame_count - self.last_rep_frame
                 min_frames = self.min_rep_frames
                 
                 if frames_since_last > min_frames:
                     self.reps += 1
                     self.last_rep_frame = self.frame_count
                     
                     # --------------------------------------------------
                     # NEW: Calculate Rep Metrics
                     # --------------------------------------------------
                     # 1. Tempo (Duration)
                     rep_duration = (self.frame_count - self.current_rep_start_frame) / self.curr_fps
                     if rep_duration > 0 and rep_duration < 10: # Filter outliers
                        self.rep_durations.append(round(rep_duration, 2))
                     
                     # 2. Range of Motion (ROM)
                     # ROM % = (Designated Start - Actual Min) / (Designated Start - Designated End)
                     # Simplified: How close did we get to the target 'up' angle?
                     # Target 'up' is the flexed state (lowest angle).
                     target_depth = cfg["up"] 
                     start_pos = cfg["down"]
                     
                     # If we went deeper than target, it's 100%+
                     # If we stayed above target, it's < 100%
                     total_range = start_pos - target_depth
                     if total_range > 0:
                        achieved_range = start_pos - self.current_rep_min_angle
                        rom_pct = (achieved_range / total_range) * 100
                        rom_pct = max(0, min(120, rom_pct)) # Cap at 0-120%
                        self.rep_roms.append(round(rom_pct, 1))

                     # CLASSIFY REP (Existing Logic)
                     if self.current_rep_forms:
                         valid_forms = [f for f in self.current_rep_forms if f != "unknown"]
                         if not valid_forms: valid_forms = ["unknown"]
                         most_common_form = Counter(valid_forms).most_common(1)[0][0]
                         if most_common_form == "correct":
                             self.reps_correct += 1
                             self.last_rep_status = "correct"
                         elif most_common_form == "wrong":
                             self.reps_wrong += 1
                             self.last_rep_status = "wrong"
                         else:
                             self.last_rep_status = "unknown"
                     else:
                         self.last_rep_status = "unknown"
                         
                     # Reset for next rep
                     self.current_rep_forms = []
                     self.current_rep_min_angle = 180 # Reset depth tracker

            self.stage = "up"
            self.current_rep_max_angle = max(self.current_rep_max_angle, angle)

        if angle < cfg["up"]: # Flexed / Squatting (Start of rep / Active phase)
            if self.stage == "up":
                # Just started the rep
                self.current_rep_start_frame = self.frame_count
                self.current_rep_min_angle = angle
            
            self.stage = "down"
            self.current_rep_min_angle = min(self.current_rep_min_angle, angle)
            
        # Track angles continuously for this rep
        if self.stage == "down":
             self.current_rep_min_angle = min(self.current_rep_min_angle, angle)

    def predict_exercise_and_form(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.pose is None:
            print("DEBUG: Pose model not loaded.")
            return "unknown", "unknown", frame, 0.0, None

        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            print("DEBUG: No pose landmarks detected.")
            return "unknown", "unknown", frame, 0.0, None

        annotated = frame.copy()
        mp_draw.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        features = extract_features(results.pose_landmarks)
        
        # --------------------------------------------------
        # NEW: Stability Tracking (Hip Center Variance)
        # --------------------------------------------------
        # Hip center x is roughly average of left (23) and right (24) hip x
        try:
             # Landmarks 23 and 24 are hips
             # Index in features (flat list): 23*4, 24*4
             # But features list is [x,y,z,v, x,y,z,v...]
             # extract_features returns 33 landmarks * 4 values
             
             # Get raw landmarks for better precision
             left_hip = results.pose_landmarks.landmark[23]
             right_hip = results.pose_landmarks.landmark[24]
             hip_center_x = (left_hip.x + right_hip.x) / 2.0
             
             self.hip_x_history.append(hip_center_x)
             
             if len(self.hip_x_history) >= 10:
                 variance = np.var(list(self.hip_x_history))
                 # Stability Score: 100 - (variance * scaling_factor)
                 # Variance usually 0.0001 (super stable) to 0.01 (shaky)
                 score = max(0, 100 - (variance * 10000))
                 self.stability_scores.append(score)
        except Exception:
             pass

        X_df = create_feature_dataframe(features, self.expected_features)
        
        # DEBUG: Check if features are empty/zeros
        if X_df.iloc[0].sum() == 0:
             print("DEBUG: CRITICAL - All features are zero! Reindexing mismatch likely.")
        
        X_scaled = self.scaler.transform(X_df)

        # ROBUST PREDICTION: Bypass sklearn wrapper to avoid version attribute errors
        try:
            dmat = xgb.DMatrix(X_scaled)
            probs = self.model.get_booster().predict(dmat)[0]
            
            # DEBUG: Print Top 3
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3_labels = self.label_encoder.inverse_transform(top3_idx)
            top3_probs = probs[top3_idx]
            print(f"DEBUG: Top Predictions: {list(zip(top3_labels, top3_probs))}")
            
        except Exception as e:
            print(f"DEBUG: XGB Prediction Error: {e}")
            return "unknown", "unknown", frame, 0.0, None

        idx = int(np.argmax(probs))
        confidence = float(np.max(probs) * 100)

        label = self.label_encoder.inverse_transform([idx])[0]
        print(f"DEBUG: Frame Pred: {label} ({confidence:.1f}%)")

        if "_" in label:
            exercise, form = label.rsplit("_", 1)
        else:
            exercise, form = label, "unknown"

        # ============================================================
        # FIX 1: Skip warm-up frames (early frames are often noisy)
        # ============================================================
        if self.frame_count <= self.WARMUP_FRAMES:
            return exercise, form, annotated, confidence, results.pose_landmarks

        # ============================================================
        # FIX 2: Only accept predictions above confidence threshold
        # ============================================================
        if confidence < self.MIN_CONFIDENCE:
            print(f"DEBUG: Low confidence {confidence:.1f}% < {self.MIN_CONFIDENCE}% - SKIPPING")
            # Still return landmarks for heatmap but don't pollute history
            avg_conf = float(np.mean(self.confidence_history)) if self.confidence_history else 0
            prev_ex = self.current_exercise if self.current_exercise != "unknown" else exercise
            prev_form = self.current_form if self.current_form != "unknown" else form
            return prev_ex, prev_form, annotated, avg_conf, results.pose_landmarks

        # ============================================================
        # FIX 3: Exercise stability lock - prevent flip-flopping
        # ============================================================
        if self.locked_exercise is not None:
            # Exercise is locked - only accept same exercise predictions
            if exercise != self.locked_exercise:
                # Different exercise detected, but we're locked
                # Allow override ONLY if confidence is very high and sustained
                if confidence > PredictionConfig.HIGH_CONFIDENCE:
                    self.lock_count -= PredictionConfig.LOCK_EROSION_RATE  # Erode lock
                    if self.lock_count <= 0:
                        print(f"DEBUG: LOCK BROKEN - Switching from {self.locked_exercise} to {exercise}")
                        self.locked_exercise = exercise
                        self.lock_count = 5
                else:
                    # Keep locked exercise, ignore this frame's exercise
                    exercise = self.locked_exercise
            else:
                self.lock_count = min(self.lock_count + 1, self.LOCK_THRESHOLD * 2)
        else:
            # No lock yet - check if we should lock
            self.prediction_history.append((exercise, form))
            self.confidence_history.append(confidence)
            
            # Count consecutive same-exercise predictions
            recent_exercises = [p[0] for p in self.prediction_history]
            if len(recent_exercises) >= self.LOCK_THRESHOLD:
                counts = Counter(recent_exercises[-self.LOCK_THRESHOLD:])
                top_exercise, top_count = counts.most_common(1)[0]
                if top_count >= self.LOCK_THRESHOLD * PredictionConfig.LOCK_AGREEMENT:
                    self.locked_exercise = top_exercise
                    self.lock_count = top_count
                    print(f"DEBUG: EXERCISE LOCKED -> {self.locked_exercise} ({top_count}/{self.LOCK_THRESHOLD})")

        # Add to history (after potential lock adjustment)
        self.prediction_history.append((exercise, form))
        self.confidence_history.append(confidence)

        # ============================================================
        # FIX 4: Smoothed prediction using majority vote from history
        # ============================================================
        exercise, form = max(
            set(self.prediction_history),
            key=self.prediction_history.count
        )

        avg_conf = float(np.mean(self.confidence_history))

        return exercise, form, annotated, avg_conf, results.pose_landmarks

    def process_frame(self, frame, return_dual=False):
        self.frame_count += 1
        exercise, form, frame_skeleton, conf, landmarks = self.predict_exercise_and_form(frame)

        if landmarks:
            self.current_exercise = exercise
            self.current_form = form
            self.heatmap_viz.update_heatmap(landmarks.landmark, frame.shape)
            self.update_reps(landmarks, exercise, current_form=form)
        else:
            self.no_pose_frames += 1

        # Generate Heatmap Frame (New Copy) - must be done AFTER skeleton frame is ready
        frame_heatmap = self.heatmap_viz.apply_heatmap_overlay(frame_skeleton.copy())

        # 1. Info Panel for NORMAL (Heatmap: OFF)
        panel_normal = create_compact_info_panel(
            frame_skeleton.shape[1],
            exercise,
            form,
            conf,
            self.stage,
            COLORS,
            self.reps,
            self.hold_time,
            show_heatmap=False,
            frame_count=self.frame_count,
            last_rep_status=self.last_rep_status,
            reps_correct=self.reps_correct,
            reps_wrong=self.reps_wrong
        )
        
        # Safe Resize to ensure exact width match (OpenCV vstack strict requirement)
        if panel_normal.shape[1] != frame_skeleton.shape[1]:
             panel_normal = cv2.resize(panel_normal, (frame_skeleton.shape[1], panel_normal.shape[0]))
        
        # Ensure both have same number of channels
        if panel_normal.shape[2] != frame_skeleton.shape[2]:
            if frame_skeleton.shape[2] == 3 and panel_normal.shape[2] == 1:
                panel_normal = cv2.cvtColor(panel_normal, cv2.COLOR_GRAY2BGR)
            elif frame_skeleton.shape[2] == 1 and panel_normal.shape[2] == 3:
                frame_skeleton = cv2.cvtColor(frame_skeleton, cv2.COLOR_GRAY2BGR)
        
        try:
             final_normal = np.vstack([panel_normal, frame_skeleton])
        except Exception as e:
             # Fallback if vstack fails (e.g. channel mismatch)
             print(f"Warning: vstack failed for normal frame: {e}")
             final_normal = frame_skeleton

        # 2. Info Panel for HEATMAP (Heatmap: ON)
        panel_heatmap = create_compact_info_panel(
            frame_heatmap.shape[1],
            exercise,
            form,
            conf,
            self.stage,
            COLORS,
            self.reps,
            self.hold_time,
            show_heatmap=True,
            frame_count=self.frame_count,
            last_rep_status=self.last_rep_status,
            reps_correct=self.reps_correct,
            reps_wrong=self.reps_wrong
        )
        
        if panel_heatmap.shape[1] != frame_heatmap.shape[1]:
             panel_heatmap = cv2.resize(panel_heatmap, (frame_heatmap.shape[1], panel_heatmap.shape[0]))

        # Ensure both have same number of channels
        if panel_heatmap.shape[2] != frame_heatmap.shape[2]:
            if frame_heatmap.shape[2] == 3 and panel_heatmap.shape[2] == 1:
                panel_heatmap = cv2.cvtColor(panel_heatmap, cv2.COLOR_GRAY2BGR)
            elif frame_heatmap.shape[2] == 1 and panel_heatmap.shape[2] == 3:
                frame_heatmap = cv2.cvtColor(frame_heatmap, cv2.COLOR_GRAY2BGR)

        try:
             final_heatmap = np.vstack([panel_heatmap, frame_heatmap])
        except Exception as e:
             print(f"Warning: vstack failed for heatmap frame: {e}")
             final_heatmap = frame_heatmap

        if exercise != "unknown":
            self.exercise_counts[exercise] = self.exercise_counts.get(exercise, 0) + 1
            # FIX: Confidence-weighted scoring (high confidence = more weight)
            weight = max(conf / 100.0, 0.1)  # Normalize to 0.1-1.0
            self.exercise_weighted_scores[exercise] = self.exercise_weighted_scores.get(exercise, 0) + weight
            # FIX: Track form PER exercise
            if exercise not in self.exercise_form_counts:
                self.exercise_form_counts[exercise] = {"correct": 0, "wrong": 0, "unknown": 0}
            if form in self.exercise_form_counts[exercise]:
                self.exercise_form_counts[exercise][form] += 1
            self.form_counts[form] += 1
            self.predictions.append((exercise, form, conf))

        if return_dual:
            return final_normal, final_heatmap

        # Legacy return if not dual
        return final_heatmap if self.heatmap_viz.show_heatmap else final_normal

    def process_video(self, input_path, output_dir="predicted_videos", enable_heatmap=True):
        self.reset_state()
        self.heatmap_viz.reset() # Reset heatmap state to avoid dimension mismatch/ghosting
        self.heatmap_viz.show_heatmap = True # Always enable internally to track heatmap state
        
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30 # Fallback
        self.curr_fps = fps
        self.min_rep_frames = int(self.MIN_REP_TIME * fps)
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ret, first = cap.read()
        if not ret:
            return {"error": "Could not read video"}

        panel = create_compact_info_panel(
            first.shape[1], "unknown", "unknown", 0.0,
            "down", COLORS, 0, 0, False, 0
        )
        h = panel.shape[0] + first.shape[0]
        w = first.shape[1]
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        os.makedirs(output_dir, exist_ok=True)
        
        uuid_str = uuid.uuid4().hex
        filename_normal = f"processed_{uuid_str}_normal.mp4"
        filename_heatmap = f"processed_{uuid_str}_heatmap.mp4"
        
        path_normal = os.path.join(output_dir, filename_normal)
        path_heatmap = os.path.join(output_dir, filename_heatmap)
        
        # Use H.264 (avc1) for better compatibility with Flutter/Web
        try:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out_normal = cv2.VideoWriter(path_normal, fourcc, fps, (w, h))
            out_heatmap = cv2.VideoWriter(path_heatmap, fourcc, fps, (w, h))
        except Exception:
            # Fallback to mp4v if avc1 is not available
            print("Warning: avc1 codec not found, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_normal = cv2.VideoWriter(path_normal, fourcc, fps, (w, h))
            out_heatmap = cv2.VideoWriter(path_heatmap, fourcc, fps, (w, h))

        count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                count += 1
                f_normal, f_heatmap = self.process_frame(frame, return_dual=True)
                
                f_normal = cv2.resize(f_normal, (w, h))
                f_heatmap = cv2.resize(f_heatmap, (w, h))
                
                draw_progress_bar(f_normal, count, total)
                draw_progress_bar(f_heatmap, count, total)
                
                out_normal.write(f_normal)
                out_heatmap.write(f_heatmap)
                
                if count % 10 == 0:
                    print(f"Processing: {int(count/total*100)}% ({count}/{total})", end='\r')
        finally:
            cap.release()
            out_normal.release()
            out_heatmap.release()
        
        # Calculate Final Stats
        # Human Presence Validation
        valid_frames = self.frame_count - self.no_pose_frames
        print(f"DEBUG: Valid Frames with Human Pose: {valid_frames}/{self.frame_count}")

        # REJECTION LOGIC: If fewer than MIN_VALID_FRAMES had a detected person, reject the video
        if self.frame_count > 0 and valid_frames < PredictionConfig.MIN_VALID_FRAMES:
             print(f"REJECTING VIDEO: Insufficient pose detection ({valid_frames}/{self.frame_count} frames).")
             # Clean up files
             try:
                 if os.path.exists(path_normal):
                     os.remove(path_normal)
                 if os.path.exists(path_heatmap):
                     os.remove(path_heatmap)
             except Exception as cleanup_err:
                 print(f"Warning: Failed to cleanup rejected video files: {cleanup_err}")

             return {"error": "No human detected in the video. Please ensure a person is fully visible for accurate analysis."}

        # ============================================================
        # IMPROVED FINAL RESULT: Use confidence-weighted voting
        # ============================================================
        final_exercise = "unknown"
        
        # Priority 1: Use locked exercise if available (most stable)
        if self.locked_exercise and self.locked_exercise in self.exercise_counts:
            final_exercise = self.locked_exercise
            print(f"DEBUG: Final exercise from LOCK: {final_exercise}")
        # Priority 2: Use confidence-WEIGHTED scores (not raw counts)
        elif self.exercise_weighted_scores:
            known_scores = {k: v for k, v in self.exercise_weighted_scores.items() if k != "unknown"}
            if known_scores:
                final_exercise = max(known_scores, key=known_scores.get)
            elif self.exercise_counts:
                final_exercise = max(self.exercise_counts, key=self.exercise_counts.get)
            print(f"DEBUG: Final exercise from WEIGHTED votes: {final_exercise}")
            # Debug: Show all weighted scores
            print(f"DEBUG: Weighted scores: {self.exercise_weighted_scores}")
        elif self.exercise_counts:
            known_exercises = {k: v for k, v in self.exercise_counts.items() if k != "unknown"}
            if known_exercises:
                final_exercise = max(known_exercises, key=known_exercises.get)
            else:
                final_exercise = max(self.exercise_counts, key=self.exercise_counts.get)
        
        # ============================================================
        # FIX: Get form for the DETECTED exercise only (not globally)
        # ============================================================
        final_form = "unknown"
        if final_exercise in self.exercise_form_counts:
            ex_forms = self.exercise_form_counts[final_exercise]
            # Filter out "unknown"
            known_forms = {k: v for k, v in ex_forms.items() if k != "unknown" and v > 0}
            if known_forms:
                final_form = max(known_forms, key=known_forms.get)
            print(f"DEBUG: Form counts for '{final_exercise}': {ex_forms} -> final: {final_form}")
        elif self.form_counts:
            # Fallback to global
            final_form = max(self.form_counts, key=self.form_counts.get)

        reco_data = load_recommendations(str(self.recommendations_file))
        final_recommendations = reco_data.get(
            final_exercise.lower(), {}
        ).get(final_form.lower(), [])

        # Calculate confidence only from predictions of the FINAL exercise
        exercise_preds = [p for p in self.predictions if p[0] == final_exercise]
        avg_conf = (
            sum(p[2] for p in exercise_preds) / len(exercise_preds)
            if exercise_preds else
            (sum(p[2] for p in self.predictions) / len(self.predictions) if self.predictions else 0)
        )
        
        # Determine confidence level for frontend
        if avg_conf >= PredictionConfig.CONFIDENCE_HIGH:
            confidence_level = "high"
        elif avg_conf >= PredictionConfig.CONFIDENCE_MEDIUM:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Assess video quality and provide recommendations
        quality_assessment = QualityMetrics.assess_video_quality(
            self.predictions, self.frame_count, valid_frames
        )
        
        # Calculate Averages for Advanced Metrics
        avg_rom = np.mean(self.rep_roms) if self.rep_roms else 0.0
        avg_tempo = np.mean(self.rep_durations) if self.rep_durations else 0.0
        avg_stability = np.mean(self.stability_scores) if self.stability_scores else 0.0

        return {
            "success": True,
            "exercise": final_exercise,
            "form": final_form,
            "reps": self.reps,
            "reps_correct": self.reps_correct,
            "reps_wrong": self.reps_wrong,
            "hold_time": self.hold_time,
            "confidence": avg_conf,
            "confidence_level": confidence_level,
            "total_frames": self.frame_count,
            "no_pose_frames": self.no_pose_frames,
            "recommendations": final_recommendations,
            "exercise_scores": self.exercise_weighted_scores,
            "processed_video_filename": filename_normal,
            "processed_video_path": path_normal,
            "video_filename_normal": filename_normal,
            "video_path_normal": path_normal,
            "video_filename_heatmap": filename_heatmap,
            "video_path_heatmap": path_heatmap,
            "quality": quality_assessment["quality"],
            "quality_issues": quality_assessment["issues"],
            "quality_recommendations": quality_assessment["recommendations"],
            
            # NEW: Advanced Analytics Data
            "advanced_metrics": {
                "avg_rom": round(avg_rom, 1),
                "avg_tempo": round(avg_tempo, 2),
                "avg_stability": round(avg_stability, 1),
                "rep_durations": self.rep_durations,
                "rep_roms": self.rep_roms
            }
        }
# Global instance to load models once
_processor = None

def get_processor():
    global _processor
    if _processor is None:
        _processor = FitnessVideoProcessor()
    return _processor
