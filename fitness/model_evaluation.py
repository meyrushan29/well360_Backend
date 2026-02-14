import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

# ----------------- PATHS -----------------
DATA_PATH = "exercise_dataset_with_phase.csv"

MODEL_PATH = "Models/exercise_form_detector.pkl"
SCALER_PATH = "Models/scaler.pkl"
ENCODER_PATH = "Models/label_encoder.pkl"
FEATURES_PATH = "Models/training_features.pkl"

# ----------------- LOAD SAVED OBJECTS -----------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)
features = joblib.load(FEATURES_PATH)

# ----------------- LOAD & PREPROCESS DATA -----------------
df = pd.read_csv(DATA_PATH).dropna()

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

df["combined"] = df["exercise_type"] + "_" + df["label"].map({1: "correct", 0: "wrong"})

X = df[features]
y_true = encoder.transform(df["combined"])

# ----------------- SCALE FEATURES -----------------
X_scaled = scaler.transform(X)

# ----------------- PREDICTIONS -----------------
y_pred = model.predict(X_scaled)

# ----------------- METRICS -----------------
accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro")
f1_weighted = f1_score(y_true, y_pred, average="weighted")

print("\n📊 FINAL MODEL PERFORMANCE")
print("=" * 50)
print(f"✅ Accuracy        : {accuracy:.4f}")
print(f"✅ F1 Score (Macro): {f1_macro:.4f}")
print(f"✅ F1 Score (Weighted): {f1_weighted:.4f}")

print("\n📌 CLASS-WISE METRICS")
print("=" * 50)
print(
    classification_report(
        encoder.inverse_transform(y_true),
        encoder.inverse_transform(y_pred)
    )
)
