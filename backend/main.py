import json

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (DisasterInput, DisasterPrediction, ImagePrediction,
                      Preparedness, PreparednessRequest, RegionRisk)
from .ml.paths import DISASTER_METRICS, VISION_METRICS
from .services import disaster, history, preparedness, regions, vision

app = FastAPI(title="FireWatch API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ok", "mock": True}


@app.post("/api/predict/image", response_model=ImagePrediction)
async def predict_image(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Upload an image file")
    try:
        return vision.predict_image(await file.read())
    except Exception:
        raise HTTPException(400, "Could not read image")


@app.post("/api/predict/disaster", response_model=DisasterPrediction)
def predict_disaster(body: DisasterInput):
    try:
        return disaster.predict_disaster(body)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/api/preparedness", response_model=Preparedness)
def get_preparedness(body: PreparednessRequest):
    return preparedness.generate_preparedness(body)


@app.get("/api/regions/risk", response_model=list[RegionRisk])
def region_risk():
    return regions.region_risks()


@app.get("/api/regions/history")
def region_history(year: int):
    try:
        return history.events_for_year(year)
    except Exception as e:
        raise HTTPException(502, f"NASA EONET request failed: {e}")


@app.get("/api/model-info")
def model_info():
    load = lambda p: json.loads(p.read_text()) if p.exists() else None
    return {"vision": load(VISION_METRICS), "disaster": load(DISASTER_METRICS)}
