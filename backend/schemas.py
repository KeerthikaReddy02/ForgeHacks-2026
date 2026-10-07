from typing import Optional, Union

from pydantic import BaseModel, Field


class ImagePrediction(BaseModel):
    label: str  # "Wildfire" | "No Wildfire"
    confidence: float
    heatmap_png_b64: str  # Grad-CAM overlay, base64 PNG
    mock: bool = True


class DisasterInput(BaseModel):
    # every measurement is optional: leaving one out means "not observed", which the model uses
    region: str = ""
    temperature_c: Optional[float] = None
    rainfall_mm: Optional[float] = None
    humidity_pct: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    magnitude: Optional[float] = None
    depth_km: Optional[float] = None
    description: str = ""
    features: dict[str, Union[float, str]] = Field(default_factory=dict)  # exact dataset columns


class ClassProb(BaseModel):
    label: str
    probability: float


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class DisasterPrediction(BaseModel):
    top3: list[ClassProb]
    top_features: list[FeatureContribution]
    mock: bool = True


class PreparednessRequest(BaseModel):
    image_result: Optional[ImagePrediction] = None
    disaster_result: Optional[DisasterPrediction] = None
    region: str = ""


class ChecklistSection(BaseModel):
    title: str
    items: list[str] = Field(default_factory=list)


class Preparedness(BaseModel):
    explanation: str
    checklist: list[ChecklistSection]
    mock: bool = True


class RegionRisk(BaseModel):
    region: str
    lat: float
    lon: float
    risk: float  # 0..1
