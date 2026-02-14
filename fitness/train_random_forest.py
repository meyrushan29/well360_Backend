import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ===============================
# DIRECTORIES
# ===============================
MODEL_DIR = "Models_RF"
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# TRAINER CLASS
# ===============================
class ExerciseFormDetectorRF:

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.features = None

    # -----------------------------
    def load_data(self, path):
        df = pd.read_csv(path).dropna()

        # Normalize exercise names
        df["exercise_type"] = (
            df["exercise_type"]
            .str.lower()
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )

        # Drop unused columns
        if "frame" in df.columns:
            df.drop(columns=["frame"], inplace=True)
        if "phase" in df.columns:
            df.drop(columns=["phase"], inplace=True)

        # Combine label
        df["combined"] = (
            df["exercise_type"] + "_" +
            df["label"].map({1: "correct", 0: "wrong"})
        )

        EXCLUDE = ["exercise_type", "label", "combined"]
        self.features = [c for c in df.columns if c not in EXCLUDE]

        X = df[self.features]
        y = self.encoder.fit_transform(df["combined"])

        return train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

    # -----------------------------
    def train(self, Xtr, Xte, ytr, yte):
        Xtr = self.scaler.fit_transform(Xtr)
        Xte = self.scaler.transform(Xte)

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(Xtr, ytr)

        preds = self.model.predict(Xte)

        print("\n📊 RANDOM FOREST CLASSIFICATION REPORT\n")
        print(
            classification_report(
                self.encoder.inverse_transform(yte),
                self.encoder.inverse_transform(preds)
            )
        )

    # -----------------------------
    def save_feature_importance(self):
        importances = self.model.feature_importances_

        df_imp = pd.DataFrame({
            "feature": self.features,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        df_imp.to_csv(
            f"{MODEL_DIR}/feature_importance_rf.csv",
            index=False
        )

        df_imp.to_json(
            f"{MODEL_DIR}/feature_importance_rf.json",
            orient="records",
            indent=4
        )

        print("📊 Feature importance saved")

    # -----------------------------
    def save(self):
        joblib.dump(
            self.model,
            f"{MODEL_DIR}/random_forest_exercise_form.pkl"
        )
        joblib.dump(
            self.scaler,
            f"{MODEL_DIR}/scaler.pkl"
        )
        joblib.dump(
            self.encoder,
            f"{MODEL_DIR}/label_encoder.pkl"
        )
        joblib.dump(
            self.features,
            f"{MODEL_DIR}/training_features.pkl"
        )

        self.save_feature_importance()

        print(f"\n✅ Random Forest model saved in `{MODEL_DIR}/`")

# ===============================
# RUN TRAINING
# ===============================
if __name__ == "__main__":
    DATA_PATH = "exercise_dataset_with_phase.csv"

    detector = ExerciseFormDetectorRF()
    Xtr, Xte, ytr, yte = detector.load_data(DATA_PATH)
    detector.train(Xtr, Xte, ytr, yte)
    detector.save()
