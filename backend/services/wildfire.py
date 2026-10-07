from ..ml.paths import WILDFIRE_MODEL,WILDFIRE_EXPLAINER


from ..ml.wildfire_engine import WildfireEngine

from ..schemas import WildfireInput, WildfirePrediction


# Keep one engine in memory.
_engine = None


def get_wildfire_engine():

    global _engine
    if _engine is None:

        _engine = WildfireEngine(model_path=WILDFIRE_MODEL,explainer_path=WILDFIRE_EXPLAINER)
    return _engine


def predict_wildfire(body: WildfireInput) -> WildfirePrediction:
    engine = get_wildfire_engine()
    return engine.predict(body)