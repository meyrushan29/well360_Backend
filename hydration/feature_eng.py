import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

from core.config import TIME_SLOT_MAPPING, ACTIVITY_MAPPING, SWEATING_MAPPING
from core.utils import setup_logging

LOG = setup_logging()


# ======================================================
# ADVANCED FEATURE ENGINEER
# (TIME-WINDOW AWARE · PREDICTION SAFE)
# ======================================================
class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.feature_names: List[str] = []

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = self._ensure_data_types(X)

        # --------------------------------------------------
        # DROP IDENTIFIERS
        # --------------------------------------------------
        X.drop(columns=[c for c in ["UserID", "Date"] if c in X.columns], inplace=True)

        # --------------------------------------------------
        # TIME WINDOW → CIRCADIAN FACTOR (CORE LOGIC)
        # --------------------------------------------------
        if "Time Slot (Select Your Current 4-Hour Window)" in X.columns:
            X["Time_Slot_Encoded"] = X[
                "Time Slot (Select Your Current 4-Hour Window)"
            ].map(lambda v: TIME_SLOT_MAPPING.get(str(v).strip(), 2))
        else:
            # Prediction-safe default (daytime)
            X["Time_Slot_Encoded"] = 2

        X["Circadian_Factor"] = np.select(
            [
                X["Time_Slot_Encoded"].isin([1, 2]),  # Morning
                X["Time_Slot_Encoded"] == 3,          # Afternoon
                X["Time_Slot_Encoded"] == 4,          # Evening
                X["Time_Slot_Encoded"].isin([0, 5])   # Night
            ],
            [1.1, 1.3, 1.0, 0.8],
            default=1.0
        )

        # --------------------------------------------------
        # BODY METRICS
        # --------------------------------------------------
        X["BMI"] = X["Weight"] / ((X["Height"] / 100) ** 2)
        X["BSA"] = np.sqrt((X["Height"] * X["Weight"]) / 3600)

        # --------------------------------------------------
        # HYDRATION INDEX (ml/kg)
        # --------------------------------------------------
        X["Hydration_Index"] = (X["Water_Intake_Last_4_Hours"] * 1000) / X["Weight"]

        # --------------------------------------------------
        # ACTIVITY & SWEATING FACTORS
        # --------------------------------------------------
        X["Activity_Factor"] = X["Physical_Activity_Level"].map(
            lambda v: ACTIVITY_MAPPING.get(str(v).strip(), 1.2)
        )

        X["Sweating_Factor"] = X["Sweating Level (Last 4 Hours)"].map(
            lambda v: SWEATING_MAPPING.get(str(v).strip(), 1)
        )

        # --------------------------------------------------
        # URINE HEALTH SCORE
        # --------------------------------------------------
        urine = X["Urine Color (Most Recent Urination)"].clip(1, 8)
        X["Urine_Health_Score"] = np.where(urine <= 3, 10 - urine, 0)

        # --------------------------------------------------
        # SYMPTOM SCORE
        # --------------------------------------------------
        symptom_cols = [
            "Thirsty (Right Now)",
            "Dizziness (Right Now)",
            "Fatigue / Tiredness (Right Now)",
            "Headache (Right Now)"
        ]

        X["Total_Symptom_Score"] = sum(
            X[col].astype(str).str.lower().eq("yes").astype(int)
            for col in symptom_cols if col in X.columns
        )

        # --------------------------------------------------
        # MEDICAL RISK FLAG (OPTIONAL – SAFE)
        # --------------------------------------------------
        if "Existing Diseases / Medical Conditions" in X.columns:
            X["Medical_Risk_Flag"] = (
                ~X["Existing Diseases / Medical Conditions"]
                .astype(str)
                .str.lower()
                .isin(["none", "unknown", ""])
            ).astype(int)
        else:
            # Prediction-time default
            X["Medical_Risk_Flag"] = 0

        # --------------------------------------------------
        # HEAT INDEX
        # --------------------------------------------------
        # --------------------------------------------------
        # HEAT INDEX (Corrected for C -> F -> C)
        # --------------------------------------------------
        # Convert C to F
        T = X["Temperature_C"] * 1.8 + 32
        R = X["Humidity_%"]

        # NOAA Regression Equation (Valid for T > 80F, RH > 40%)
        # For simplicity, we apply it uniformly as the model was trained this way.
        
        c1 = -42.379
        c2 = 2.04901523
        c3 = 10.14333127
        c4 = -0.22475541
        c5 = -0.00683783
        c6 = -0.05481717
        c7 = 0.00122874
        c8 = 0.00085282
        c9 = -0.00000199

        HI_f = (c1 + (c2 * T) + (c3 * R) + (c4 * T * R) + 
               (c5 * T**2) + (c6 * R**2) + 
               (c7 * T**2 * R) + (c8 * T * R**2) + 
               (c9 * T**2 * R**2))

        # Convert back to C
        X["Heat_Index"] = (HI_f - 32) / 1.8

        # --------------------------------------------------
        # WATER DEFICIT (NEXT 4 HOURS)
        # --------------------------------------------------
        # Baseline needs: ~0.03 L/kg per day -> per 4 hours (/6)
        baseline_need = (X["Weight"] * 0.033) / 6

        # Exercise loss: approx 0.5L - 1.5L per hour depending on intensity
        # We use Activity_Factor as a proxy for intensity multiplier
        exercise_loss = (X["Exercise Time (minutes) in Last 4 Hours"] / 60) * (X["Activity_Factor"] * 0.4)

        # Sweating loss (additional):
        sweat_loss = X["Sweating_Factor"] * 0.15  # approx 150ml per level of sweat

        total_need = baseline_need + exercise_loss + sweat_loss
        X["Water_Deficit"] = (total_need - X["Water_Intake_Last_4_Hours"]).clip(lower=0)

        # --------------------------------------------------
        # COMPOSITE HYDRATION SCORE (TIME-AWARE)
        # --------------------------------------------------
        X["Composite_Hydration_Score"] = (
            X["Hydration_Index"] * 0.25 +
            X["Urine_Health_Score"] * 0.20 +
            (4 - X["Total_Symptom_Score"].clip(0, 4)) * 0.20 +
            X["Activity_Factor"] * 0.15 +
            (1 - X["Medical_Risk_Flag"]) * 0.10 +
            X["Circadian_Factor"] * 0.10
        )

        self.feature_names = X.columns.tolist()
        LOG.info(f"Feature engineering completed | Features: {len(self.feature_names)}")

        return X

    # --------------------------------------------------
    # TYPE SAFETY
    # --------------------------------------------------
    def _ensure_data_types(self, X: pd.DataFrame) -> pd.DataFrame:

        numeric_cols = [
            "Age", "Weight", "Height",
            "Exercise Time (minutes) in Last 4 Hours",
            "Water_Intake_Last_4_Hours",
            "Temperature_C", "Humidity_%",
            "Urine Color (Most Recent Urination)"
        ]

        for col in numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].fillna(X[col].median())

        return X


# ======================================================
# HELPER FUNCTION
# ======================================================
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    return AdvancedFeatureEngineer().transform(df)
