import cv2
import joblib
import mediapipe as mp
import numpy as np
import os
import time
from collections import deque

from exercise_config import EXERCISE_CONFIG, COLORS
from heatmap import HeatmapVisualizer
from geometry import calculate_angles_from_landmarks
from utils import (
    create_compact_info_panel,
    draw_progress_bar,
    extract_features,
    create_feature_dataframe,
    find_video_files,
    load_recommendations,
    save_detailed_report_with_recommendations
)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

os.makedirs("predicted_rf", exist_ok=True)

# =========================================================
# TEXT SUMMARY
# =========================================================
def save_text_summary(
    video_path,
    exercise,
    form,
    reps,
    hold_time,
    predictions,
    frame_count,
    no_pose_frames,
    recommendations
):
    summary_path = f"predicted_rf/{os.path.basename(video_path)}_summary.txt"

    avg_conf = (
        sum(p[2] for p in predictions) / len(predictions)
        if predictions else 0
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("EXERCISE FORM ANALYSIS SUMMARY (RANDOM FOREST)\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Video File        : {os.path.basename(video_path)}\n")
        f.write(f"Detected Exercise : {exercise}\n")
        f.write(f"Detected Form     : {form}\n")
        f.write(f"Average Confidence: {avg_conf:.2f}%\n\n")

        if reps > 0:
            f.write(f"Total Repetitions : {reps}\n")
        if hold_time > 0:
            f.write(f"Hold Time         : {hold_time} seconds\n")

        f.write("\nFRAME STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Frames      : {frame_count}\n")
        f.write(f"No-Pose Frames    : {no_pose_frames}\n")

        f.write("\nRECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")

        if recommendations:
            for i, tip in enumerate(recommendations, 1):
                f.write(f"{i}. {tip}\n")
        else:
            f.write("No recommendations available.\n")

    print(f"📝 RF Text summary saved: {summary_path}")


# =========================================================
# MAIN PREDICTOR (RANDOM FOREST)
# =========================================================
class ExerciseFormPredictorRF:
    def __init__(self):
        print("\n--> Loading Random Forest model components")

        self.model = joblib.load("Models_RF/random_forest_exercise_form.pkl")
        self.scaler = joblib.load("Models_RF/scaler.pkl")
        self.label_encoder = joblib.load("Models_RF/label_encoder.pkl")
        self.expected_features = joblib.load("Models_RF/training_features.pkl")

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.heatmap_viz = HeatmapVisualizer()

        self.prediction_history = deque(maxlen=15)
        self.confidence_history = deque(maxlen=15)

        self.reps = 0
        self.stage = "down"
        self.last_rep_time = 0
        self.MIN_REP_TIME = 0.8

        self.hold_time = 0
        self.hold_start = None

        self.current_exercise = "unknown"
        self.current_form = "unknown"

        self.exercise_counts = {}
        self.form_counts = {"correct": 0, "wrong": 0, "unknown": 0}
        self.predictions = []
        self.no_pose_frames = 0
        self.frame_count = 0

    # ---------------------------------------------------------
    def update_reps(self, landmarks, exercise):
        if exercise not in EXERCISE_CONFIG:
            return

        cfg = EXERCISE_CONFIG[exercise]

        if cfg["type"] == "hold":
            if self.hold_start is None:
                self.hold_start = time.time()
            self.hold_time = int(time.time() - self.hold_start)
            return

        angles = calculate_angles_from_landmarks(landmarks, cfg)
        if "main" not in angles:
            return

        angle = angles["main"]

        if angle > cfg["down"]:
            self.stage = "down"

        if angle < cfg["up"] and self.stage == "down":
            now = time.time()
            if now - self.last_rep_time > self.MIN_REP_TIME:
                self.reps += 1
                self.last_rep_time = now
                self.stage = "up"

    # ---------------------------------------------------------
    def predict_exercise_and_form(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return "unknown", "unknown", frame, 0.0, None

        annotated = frame.copy()
        mp_draw.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        features = extract_features(results.pose_landmarks)
        X_df = create_feature_dataframe(features, self.expected_features)
        X_scaled = self.scaler.transform(X_df)

        probs = self.model.predict_proba(X_scaled)[0]
        idx = int(np.argmax(probs))
        confidence = float(np.max(probs) * 100)

        label = self.label_encoder.inverse_transform([idx])[0]

        if "_" in label:
            exercise, form = label.rsplit("_", 1)
        else:
            exercise, form = label, "unknown"

        self.prediction_history.append((exercise, form))
        self.confidence_history.append(confidence)

        exercise, form = max(
            set(self.prediction_history),
            key=self.prediction_history.count
        )

        avg_conf = float(np.mean(self.confidence_history))

        return exercise, form, annotated, avg_conf, results.pose_landmarks

    # ---------------------------------------------------------
    def process_frame(self, frame):
        self.frame_count += 1

        exercise, form, frame, conf, landmarks = self.predict_exercise_and_form(frame)

        if landmarks:
            self.current_exercise = exercise
            self.current_form = form
            self.heatmap_viz.update_heatmap(landmarks.landmark, frame.shape)
            self.update_reps(landmarks, exercise)
        else:
            self.no_pose_frames += 1

        frame = self.heatmap_viz.apply_heatmap_overlay(frame)

        info_panel = create_compact_info_panel(
            frame.shape[1],
            exercise,
            form,
            conf,
            self.stage,
            COLORS,
            self.reps,
            self.hold_time,
            self.heatmap_viz.show_heatmap,
            self.frame_count
        )

        final_frame = np.vstack([info_panel, frame])

        if exercise != "unknown":
            self.exercise_counts[exercise] = self.exercise_counts.get(exercise, 0) + 1
            self.form_counts[form] += 1
            self.predictions.append((exercise, form, conf))

        return final_frame

    # ---------------------------------------------------------
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, first = cap.read()
        if not ret:
            return

        panel = create_compact_info_panel(
            first.shape[1], "unknown", "unknown", 0.0,
            "down", COLORS, 0, 0, False, 0
        )

        h = panel.shape[0] + first.shape[0]
        w = first.shape[1]

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        out_path = f"predicted_rf/output_{os.path.basename(video_path)}"
        out = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            result = self.process_frame(frame)
            result = cv2.resize(result, (w, h))

            draw_progress_bar(result, count, total)
            out.write(result)
            cv2.imshow("RF Exercise & Form Detection", result)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("h"), ord("H")):
                self.heatmap_viz.toggle()
            elif key == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        save_detailed_report_with_recommendations(
            video_path,
            self.predictions,
            self.exercise_counts,
            self.form_counts,
            self.no_pose_frames,
            count,
            self.current_exercise,
            self.reps,
            self.hold_time,
            self.label_encoder.classes_,
            load_recommendations("recommendations.json")
        )

        reco_data = load_recommendations("recommendations.json")
        final_recommendations = reco_data.get(
            self.current_exercise.lower(), {}
        ).get(self.current_form.lower(), [])

        save_text_summary(
            video_path,
            self.current_exercise,
            self.current_form,
            self.reps,
            self.hold_time,
            self.predictions,
            count,
            self.no_pose_frames,
            final_recommendations
        )

        print(f"\n✅ RF Output saved: {out_path}")


# =========================================================
def main():
    videos = find_video_files()
    for i, v in enumerate(videos, 1):
        print(f"{i}. {v}")

    idx = int(input("Select video: ")) - 1
    ExerciseFormPredictorRF().process_video(videos[idx])


if __name__ == "__main__":
    main()
