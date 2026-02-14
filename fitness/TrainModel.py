import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import json
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ----------------- DIRS -----------------
os.makedirs("Models", exist_ok=True)

# ----------------- TRAINER -----------------
class ExerciseFormDetector:

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.features = None

    def load_data(self, path):
        df = pd.read_csv(path).dropna()

        df["exercise_type"] = (
            df["exercise_type"]
            .str.lower()
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("-", "_")
        )

        if "frame" in df.columns:
            df.drop(columns=["frame"], inplace=True)
        if "phase" in df.columns:
            df.drop(columns=["phase"], inplace=True)

        df["combined"] = df["exercise_type"] + "_" + df["label"].map({1:"correct",0:"wrong"})

        EXCLUDE = ["exercise_type", "label", "combined"]
        self.features = [c for c in df.columns if c not in EXCLUDE]

        X = df[self.features]
        y = self.encoder.fit_transform(df["combined"])

        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    def train(self, Xtr, Xte, ytr, yte):
        Xtr = self.scaler.fit_transform(Xtr)
        Xte = self.scaler.transform(Xte)

        self.model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(np.unique(ytr)),
            n_estimators=300,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            random_state=42
        )

        try:
            self.model.fit(Xtr, ytr)
        except:
            self.model.set_params(tree_method="hist", predictor="auto")
            self.model.fit(Xtr, ytr)

        preds = self.model.predict(Xte)
        print(classification_report(
            self.encoder.inverse_transform(yte),
            self.encoder.inverse_transform(preds)
        ))

    def save(self):
        joblib.dump(self.model, "Models/exercise_form_detector.pkl")
        joblib.dump(self.scaler, "Models/scaler.pkl")
        joblib.dump(self.encoder, "Models/label_encoder.pkl")
        joblib.dump(self.features, "Models/training_features.pkl")

        print("✅ Model saved")

# ----------------- RUN -----------------
if __name__ == "__main__":
    data = "exercise_dataset_with_phase.csv"
    detector = ExerciseFormDetector()
    Xtr, Xte, ytr, yte = detector.load_data(data)
    detector.train(Xtr, Xte, ytr, yte)
    detector.save()
