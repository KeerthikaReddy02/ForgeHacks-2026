from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from ..schemas import WildfireInput, WildfirePrediction, RiskFactor


# MODEL FEATURES

FEATURE_COLUMNS = [
    "LATITUDE",
    "LONGITUDE",
    "MAX_TEMP",
    "PRECIPITATION",
    "WIND_SPEED",
    "MIN_HUMIDITY",
]


# RISK LEVEL

def get_risk_level(probability: float) -> str:

    if probability < 0.01:
        return "No Wildfire Risk"

    elif probability < 0.10:
        return "Very Low Risk"

    elif probability < 0.30:
        return "Low Risk"

    elif probability < 0.60:
        return "Medium Risk"

    else:
        return "High Risk"


# SHAP EXPLANATION

def get_risk_explanation(explanation, top_n=3):

    explanation = explanation.head(top_n)

    increased = explanation[
        explanation["SHAP"] > 0
    ]

    decreased = explanation[
        explanation["SHAP"] < 0
    ]

    increased_factors = []
    decreased_factors = []

    # MAIN RISK FACTORS

    for _, row in increased.iterrows():

        feature = row["Feature"]
        value = row["Value"]

        if feature == "MIN_HUMIDITY":
            description = f"Low humidity ({value}%)"

        elif feature == "MAX_TEMP":
            description = f"High temperature ({value}°C)"

        elif feature == "PRECIPITATION":
            description = f"Low precipitation ({value} mm)"

        elif feature == "WIND_SPEED":
            description = f"Wind speed ({value})"

        elif feature == "LATITUDE":
            description = f"Location latitude ({value})"

        elif feature == "LONGITUDE":
            description = f"Location longitude ({value})"

        else:
            description = f"{feature} ({value})"

        increased_factors.append({
            "feature": feature,
            "value": float(value),
            "shap": float(row["SHAP"]),
            "description": description,
            "direction": "increases",
        })

    # RISK-REDUCING FACTORS

    for _, row in decreased.iterrows():

        feature = row["Feature"]
        value = row["Value"]

        if feature == "MIN_HUMIDITY":
            description = f"Higher humidity ({value}%)"

        elif feature == "MAX_TEMP":
            description = f"Lower temperature ({value}°C)"

        elif feature == "PRECIPITATION":
            description = f"Higher precipitation ({value} mm)"

        elif feature == "WIND_SPEED":
            description = f"Wind speed ({value})"

        elif feature == "LATITUDE":
            description = f"Location latitude ({value})"

        elif feature == "LONGITUDE":
            description = f"Location longitude ({value})"

        else:
            description = f"{feature} ({value})"

        decreased_factors.append({
            "feature": feature,
            "value": float(value),
            "shap": float(row["SHAP"]),
            "description": description,
            "direction": "reduces",
        })

    return increased_factors, decreased_factors


# WILDFIRE ENGINE

class WildfireEngine:

    def __init__(
        self,
        model_path: Path,
        explainer_path: Path,
    ):

        # Load LightGBM model

        if not model_path.exists():
            raise FileNotFoundError(
                f"Wildfire model not found: {model_path}"
            )

        self.model = joblib.load(model_path)

        # Load saved SHAP explainer

        if explainer_path.exists():

            self.explainer = joblib.load(
                explainer_path
            )

        else:

            # Create the explainer only once.
            self.explainer = shap.TreeExplainer(
                self.model
            )

            # Save it for future server runs.
            joblib.dump(
                self.explainer,
                explainer_path
            )

    
    # BUILD MODEL INPUT

    def build_input(
        self,
        body: WildfireInput
    ) -> pd.DataFrame:

        user_input = pd.DataFrame([
            {
                "LATITUDE": body.latitude,
                "LONGITUDE": body.longitude,
                "MAX_TEMP": body.max_temp,
                "PRECIPITATION": body.precipitation,
                "WIND_SPEED": body.wind_speed,
                "MIN_HUMIDITY": body.min_humidity,
            }
        ])

        user_input = user_input[
            FEATURE_COLUMNS
        ]

        return user_input

    # SHAP VALUES

    def calculate_shap(
        self,
        user_input: pd.DataFrame
    ):

        # SHAP for explainability
        shap_values = self.explainer.shap_values(
            user_input
        )

        # For binary classification, depending on SHAP version,
        # shap_values may be a list or an array.

        if isinstance(shap_values, list):

            shap_for_prediction = shap_values[1][0]

        else:

            shap_for_prediction = shap_values[0]

        explanation = pd.DataFrame({
            "Feature": user_input.columns,
            "Value": user_input.iloc[0].values,
            "SHAP": shap_for_prediction
        })

        explanation["Absolute_SHAP"] = (
            explanation["SHAP"].abs()
        )

        explanation = explanation.sort_values(
            "Absolute_SHAP",
            ascending=False
        )

        return explanation

    # PREDICT

    def predict(
        self,
        body: WildfireInput
    ) -> WildfirePrediction:

        user_input = self.build_input(body)

        # Prediction probability

        probability = float(
            self.model.predict_proba(
                user_input
            )[0, 1]
        )

        risk_level = get_risk_level(
            probability
        )

        # SHAP explanation

        explanation = self.calculate_shap(
            user_input
        )

        increased, decreased = get_risk_explanation(
            explanation,
            top_n=3
        )

        # Combine them for frontend.
        top_factors = increased + decreased

        return WildfirePrediction(
            probability=probability,
            risk_level=risk_level,
            top_factors=top_factors,
            mock=False,
        )