"""Home page: showcases the app and its ways to assess risk."""
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parents[1] / "assets"

ss = st.session_state


def go(title):
    """Button callback: jump to another page (the Page objects are registered in app.py)."""
    st.switch_page(ss.pages[title])

# ---------------------------------------------------------------- hero
with st.container(key="hero"):
    hero, pic = st.columns([1, 1], gap="large", vertical_alignment="center")
    with hero:
        st.html('<p class="eyebrow">WILDFIRE PREDICTION PLATFORM</p>')
        st.title("Spot wildfire risk before it spreads.")
        st.write(
            "FireWatch checks satellite images and local conditions to estimate wildfire risk "
            "and explain what drove each prediction."
        )
        st.button(
            "Start a prediction",
            type="primary",
            icon=":material/arrow_forward:",
            icon_position="right",
            key="go_start",
        )
    with pic.container(border=True, key="hero_card"):
        st.image(str(ASSETS / "hero.jpg"), alt="Satellite view of a wildfire with a smoke plume over dry hills")
        a, b = st.columns(2)
        a.caption("Sample satellite image")
        b.markdown(":red[**Wildfire visible**]", text_alignment="right")

# ---------------------------------------------------------------- three ways
st.header("Three ways to explore risk", anchor="three-ways-to-explore-risk")
CARDS = [
    ("Satellite image analysis", ":material/satellite_alt:",
     "We use deep learning to predict whether an area is at risk of a wildfire from a satellite image.",
     "- Deep-learning image classifier\n- Label with a confidence score\n- Risk level for the whole image",
     "Analyze an image", "Satellite image"),
    ("Risk conditions form", ":material/thermostat:",
     "Enter whatever measurements you have and the model predicts the risk of wildfire.",
     "- Temperature, humidity, rainfall, wind, latitude and longitude\n- Wildfire risk with factors that influenced the decision",
     "Enter conditions", "Conditions form"),
    ("Historical risk map", ":material/public:",
     "Explore five years of wildfire activity across the world on an interactive map.",
     "- NASA EONET wildfire events, worldwide\n- Filter by year\n- Event counts and top hotspots",
     "Open the map", "Historical map"),
]
# Each button sits inside its own card; CSS in app.py pins it to the card's bottom so all three line up.
for col, (title, icon, blurb, bullets, label, target) in zip(st.columns(3, gap="medium"), CARDS):
    with col.container(border=True, height="stretch", key=f"card_{target.split()[0].lower()}"):
        st.markdown(f"##### {icon} {title}")
        st.write(blurb)
        st.markdown(bullets)
        st.button(label, type="primary", icon=":material/arrow_forward:", width="stretch",
                  key=f"go_{target}", on_click=go, args=(target,))

# ---------------------------------------------------------------- benefits
st.space("small")
f1, f3 = st.columns(2)
f1.markdown("**:green[:material/speed:] Fast results**  \nA prediction in a few seconds.", text_alignment="center")
f3.markdown("**:green[:material/verified_user:] Informational only**  \nNot a substitute for official emergency alerts.", text_alignment="center")
