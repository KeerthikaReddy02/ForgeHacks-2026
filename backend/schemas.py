from typing import Optional

from pydantic import BaseModel, Field


class ImagePrediction(BaseModel):
    label: str  # "Wildfire" | "No Wildfire"
    confidence: float
    heatmap_png_b64: str  # Grad-CAM overlay, base64 PNG
    mock: bool = True


class DisasterInput(BaseModel):
    region: str = "California"
    temperature_c: float = 30.0
    rainfall_mm: float = 10.0
    humidity_pct: float = 40.0
    description: str = ""


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
