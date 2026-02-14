import pandas as pd
import numpy as np
from pathlib import Path
from core.config import (
    DATA_PATH,
    DATA_DIR,
    TIME_SLOT_MAPPING
)
from core.utils import setup_logging

LOG = setup_logging()

# ======================================================
# CLEANING
# ======================================================
def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "Age", "Weight", "Height",
        "Exercise Time (minutes) in Last 4 Hours",
        "Water_Intake_Last_4_Hours",
        "Temperature_C", "Humidity_%"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Urine color (1–8 scale)
    if "Urine Color (Most Recent Urination)" in df.columns:
        df["Urine Color (Most Recent Urination)"] = (
            pd.to_numeric(
                df["Urine Color (Most Recent Urination)"]
                .astype(str)
                .str.extract(r"(\d+)")[0],
                errors="coerce"
            )
            .fillna(4)
            .clip(1, 8)
        )

    categorical_cols = [
        "Gender",
        "Physical_Activity_Level",
        "Urinated (Last 4 Hours)",
        "Thirsty (Right Now)",
        "Dizziness (Right Now)",
        "Fatigue / Tiredness (Right Now)",
        "Headache (Right Now)",
        "Sweating Level (Last 4 Hours)",
        "Existing Diseases / Medical Conditions"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace(["nan", "NaN", "None", "null", ""], "Unknown")
            )

    # --------------------------------------------------
    # TIME WINDOW → NUMERIC ENCODING (KEY FIX)
    # --------------------------------------------------
    if "Time Slot (Select Your Current 4-Hour Window)" in df.columns:
        df["Time Slot (Select Your Current 4-Hour Window)"] = (
            df["Time Slot (Select Your Current 4-Hour Window)"]
            .astype(str)
            .str.strip()
        )

        df["Time_Slot_Encoded"] = df[
            "Time Slot (Select Your Current 4-Hour Window)"
        ].map(lambda v: TIME_SLOT_MAPPING.get(v, 2))

    # Drop rows missing critical physiology
    df.dropna(
        subset=["Age", "Weight", "Height", "Water_Intake_Last_4_Hours"],
        inplace=True
    )

    LOG.info(f"Data cleaned | Shape: {df.shape}")
    return df


# ======================================================
# TARGET GENERATION
# ======================================================
def calculate_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Recommended_Water_Next_4_Hours"] = calculate_water_recommendation(df)
    df["Hydration_Risk_Level"] = assess_hydration_risk(df)

    # Optional / informational only
    if "Existing Diseases / Medical Conditions" in df.columns:
        df["Existing_Diseases_Predicted"] = (
            df["Existing Diseases / Medical Conditions"]
        )

    return df


# ======================================================
# WATER REQUIREMENT (NEXT 4 HOURS)
# ======================================================
def calculate_water_recommendation(df: pd.DataFrame) -> pd.Series:
    # 0.033 represents ~33ml/kg (standard daily norm)
    # Divisor 5 assumes ~5 windows of 4 hours in a waking day (20h activity cap)
    base = (df["Weight"] * 0.033) / 5

    activity_map = {
        "Sedentary": 1.0,
        "Light": 1.2,
        "Moderate": 1.5,
        "Heavy": 2.0,
        "Very Heavy": 2.5,
        "Unknown": 1.2
    }

    activity = df["Physical_Activity_Level"].map(activity_map).fillna(1.2)
    temp_adj = np.maximum(0, (df["Temperature_C"] - 25) / 5) * 0.5
    exercise_adj = df["Exercise Time (minutes) in Last 4 Hours"].fillna(0) * 0.01

    sweat_map = {
        "None": 0,
        "Light": 0.2,
        "Moderate": 0.5,
        "Heavy": 0.8,
        "Very Heavy": 1.2,
        "Unknown": 0.3
    }

    sweat_adj = (
        df["Sweating Level (Last 4 Hours)"]
        .map(sweat_map)
        .fillna(0.3)
    )

    water = base * activity + temp_adj + exercise_adj + sweat_adj
    return water.clip(0.2, 3.0).round(2)


# ======================================================
# HYDRATION RISK (STANDARDIZED LABELS)
# ======================================================
def assess_hydration_risk(df: pd.DataFrame) -> pd.Series:
    risks = []

    for _, r in df.iterrows():
        score = 0

        if r["Urine Color (Most Recent Urination)"] >= 6:
            score += 3

        symptoms = sum([
            str(r.get("Thirsty (Right Now)", "No")).lower() == "yes",
            str(r.get("Dizziness (Right Now)", "No")).lower() == "yes",
            str(r.get("Fatigue / Tiredness (Right Now)", "No")).lower() == "yes",
            str(r.get("Headache (Right Now)", "No")).lower() == "yes"
        ])
        score += symptoms * 2

        if str(r.get("Urinated (Last 4 Hours)", "Yes")).lower() == "no":
            score += 2

        expected = (r["Weight"] * 0.033) / 5
        ratio = r["Water_Intake_Last_4_Hours"] / expected if expected > 0 else 1

        if ratio < 0.5:
            score += 3
        elif ratio < 0.8:
            score += 1

        if r["Temperature_C"] > 30:
            score += 1

        if score >= 8:
            risks.append("High")
        elif score >= 5:
            risks.append("Moderate")
        elif score >= 3:
            risks.append("Low")
        else:
            risks.append("Very Low")

    return pd.Series(risks, index=df.index)


# ======================================================
# LOAD PIPELINE
# ======================================================
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    df = clean_and_prepare_data(df)
    df = calculate_targets(df)

    output = Path(DATA_DIR) / "labeled_dataset.csv"
    df.to_csv(output, index=False)

    LOG.info("Labeled dataset saved successfully")
    return df


# ======================================================
# TEST
# ======================================================
if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print("Shape:", df.shape)
