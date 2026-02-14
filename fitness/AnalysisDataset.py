import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

mp_pose = mp.solutions.pose


# ======================================================
# EXERCISE PHASE DETECTOR
# ======================================================
class ExercisePhaseDetector:
    def __init__(self, exercise_type):
        self.exercise_type = self.normalize_exercise_name(exercise_type)

    @staticmethod
    def normalize_exercise_name(name):
        if not name:
            return "unknown"
        return name.lower().strip().replace(" ", "_").replace("-", "_")

    def detect_phase(self, landmarks):
        if landmarks is None:
            return "unknown"

        k = {i: (lm.x, lm.y, lm.z) for i, lm in enumerate(landmarks.landmark)}
        et = self.exercise_type

        if any(w in et for w in ["biceps", "curl"]):
            return self._biceps(k)
        elif "squat" in et:
            return self._squat(k)
        elif any(w in et for w in ["push", "press", "dip"]):
            return self._push(k)
        elif any(w in et for w in ["pull", "row"]):
            return self._pull(k)
        elif "deadlift" in et:
            return self._deadlift(k)
        elif any(w in et for w in ["raise", "lateral"]):
            return self._raise(k)
        elif "plank" in et:
            return self._plank(k)
        elif "twist" in et:
            return self._twist(k)
        elif "leg" in et:
            return self._leg(k)
        elif "hip" in et:
            return self._hip(k)
        else:
            return "mid"

    # ---------------- PHASE LOGIC ----------------
    def _biceps(self, k):
        a = (self._angle(k[11], k[13], k[15]) +
             self._angle(k[12], k[14], k[16])) / 2
        if a < 60: return "contracted"
        if a > 150: return "extended"
        return "mid"

    def _squat(self, k):
        a = (self._angle(k[23], k[25], k[27]) +
             self._angle(k[24], k[26], k[28])) / 2
        if a < 90: return "bottom"
        if a > 160: return "top"
        return "mid"

    def _push(self, k):
        a = (self._angle(k[11], k[13], k[15]) +
             self._angle(k[12], k[14], k[16])) / 2
        if a < 80: return "bottom"
        if a > 160: return "top"
        return "mid"

    def _pull(self, k):
        a = (self._angle(k[11], k[13], k[15]) +
             self._angle(k[12], k[14], k[16])) / 2
        if a < 50: return "contracted"
        if a > 140: return "extended"
        return "mid"

    def _deadlift(self, k):
        angle = self._torso_angle(k[11], k[23])
        if angle > 45: return "bottom"
        if angle < 10: return "top"
        return "mid"

    def _raise(self, k):
        a = (self._angle(k[11], k[13], (k[13][0], k[13][1] - 0.1, k[13][2])) +
             self._angle(k[12], k[14], (k[14][0], k[14][1] - 0.1, k[14][2]))) / 2
        if a > 80: return "top"
        if a < 30: return "bottom"
        return "mid"

    def _plank(self, k):
        return "stable" if abs(self._torso_angle(k[11], k[23])) < 10 else "unstable"

    def _twist(self, k):
        rot = abs(k[11][0] - k[12][0]) - abs(k[23][0] - k[24][0])
        if rot > 0.05: return "left_twist"
        if rot < -0.05: return "right_twist"
        return "center"

    def _leg(self, k):
        a = (self._angle(k[23], k[25], k[27]) +
             self._angle(k[24], k[26], k[28])) / 2
        if a < 60: return "contracted"
        if a > 140: return "extended"
        return "mid"

    def _hip(self, k):
        h = (k[23][1] + k[24][1]) / 2
        if h < 0.3: return "top"
        if h > 0.6: return "bottom"
        return "mid"

    # ---------------- SAFE MATH ----------------
    def _angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
        if na < 1e-6 or nc < 1e-6:
            return 180.0
        cos = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
        return np.degrees(np.arccos(cos))

    def _torso_angle(self, s, h):
        dx, dy = s[0] - h[0], s[1] - h[1]
        return abs(np.degrees(np.arctan2(dx, dy)))


# ======================================================
# VIDEO → LANDMARK EXTRACTION
# ======================================================
def extract_landmarks_with_phase(video_path, exercise_type, label):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return []

    detector = ExercisePhaseDetector(exercise_type)
    data = []
    frame_no = 0

    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                row = []
                for lm in result.pose_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])

                phase = detector.detect_phase(result.pose_landmarks)

                PHASE_MAP = {
                    "top": 0, "mid": 1, "bottom": 2,
                    "extended": 3, "contracted": 4,
                    "stable": 5, "unstable": 6,
                    "left_twist": 7, "right_twist": 8,
                    "center": 9, "unknown": 10
                }

                row += [
                    exercise_type,
                    label,
                    PHASE_MAP.get(phase, 10),
                    frame_no
                ]
                data.append(row)

            frame_no += 1

    cap.release()
    return data


# ======================================================
# DATASET COLLECTOR
# ======================================================
def collect_dataset_with_phase():
    output_file = "exercise_dataset_with_phase.csv"

    columns = []
    # Pose features (132)
    for i in range(33):
        columns += [f"x_{i}", f"y_{i}", f"z_{i}", f"v_{i}"]

    # Metadata (NOT used for ML)
    columns += ["exercise_type", "label", "phase", "frame"]

    if os.path.exists(output_file):
        os.remove(output_file)

    dataset_root = "dataset"
    if not os.path.exists(dataset_root):
        print("❌ dataset folder not found")
        return

    total_videos = 0

    for exercise in os.listdir(dataset_root):
        ex_path = os.path.join(dataset_root, exercise)
        if not os.path.isdir(ex_path):
            continue

        exercise_norm = ExercisePhaseDetector.normalize_exercise_name(exercise)

        for label_name, label in [("correct", 1), ("wrong", 0)]:
            folder = os.path.join(ex_path, label_name)
            if not os.path.exists(folder):
                continue

            for file in os.listdir(folder):
                if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    print(f"▶ {exercise_norm}/{label_name}/{file}")

                    rows = extract_landmarks_with_phase(
                        os.path.join(folder, file),
                        exercise_norm,
                        label
                    )

                    if rows:
                        pd.DataFrame(rows, columns=columns).to_csv(
                            output_file,
                            mode="a",
                            header=not os.path.exists(output_file),
                            index=False
                        )

                    total_videos += 1

    print("\n✅ DATASET CREATION COMPLETE")
    print(f"📁 Output file: {output_file}")
    print(f"🎥 Videos processed: {total_videos}")


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EXERCISE DATASET COLLECTOR WITH PHASE (FINAL)")
    print("=" * 60)
    collect_dataset_with_phase()
