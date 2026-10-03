"""Preparedness generator. MOCK: swap `generate_preparedness` for an LLM call."""
from ..schemas import ChecklistSection, Preparedness, PreparednessRequest


def generate_preparedness(req: PreparednessRequest) -> Preparedness:
    hazard = req.disaster_result.top3[0].label if req.disaster_result else None
    fire = req.image_result and req.image_result.label == "Wildfire"
    where = f" in {req.region}" if req.region else ""

    parts = []
    if req.image_result:
        parts.append(
            f"The satellite image looks {'at risk of wildfire' if fire else 'low-risk for wildfire'} "
            f"({req.image_result.confidence:.0%} confidence)."
        )
    if hazard:
        parts.append(f"Current conditions{where} point most strongly to {hazard.lower()} risk.")
    explanation = " ".join(parts) or "Run a prediction to get a tailored plan."

    return Preparedness(
        explanation=explanation,
        checklist=[
            ChecklistSection(title="Evacuation prep", items=[
                "Know two evacuation routes out of your area",
                "Keep your car fuelled and parked facing out",
                "Agree on a family meeting point",
            ]),
            ChecklistSection(title="Emergency kit", items=[
                "3 days of water (4 L per person per day) and food",
                "N95 masks, first-aid kit, flashlight, batteries",
                "Copies of IDs, medications, phone charger/power bank",
            ]),
            ChecklistSection(title="Property hardening", items=[
                "Clear dry leaves and brush within 10 m of the house",
                "Clean gutters and cover vents with fine mesh",
                "Move firewood and propane away from the structure",
            ]),
            ChecklistSection(title="Resource use", items=[
                "Conserve water: shorter showers, fix leaks",
                "Avoid outdoor burning and open flames",
                "Sign up for local emergency alerts",
            ]),
        ],
    )
