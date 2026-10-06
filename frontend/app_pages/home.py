"""Home page: showcases the app and its two ways to assess risk."""
import streamlit as st

from common import cached_get

ss = st.session_state


def go(title):
    """Button callback: jump to another page (the Page objects are registered in app.py)."""
    st.switch_page(ss.pages[title])

# ---------------------------------------------------------------- hero
st.title("FireWatch", icon=":material/local_fire_department:")
st.subheader("Spot wildfire and disaster risk early, from space or from the ground.")
st.write(
    "FireWatch combines satellite imagery and on-the-ground measurements with machine learning "
    "to estimate risk and explain what drove each prediction."
)

# Headline numbers from the 5-year global fire history
hist = cached_get("/api/regions/history") or []
if hist:
    events = sum(r["events"] for r in hist)
    years = sorted({r["year"] for r in hist})
    m1, m2, m3 = st.columns(3)  # equal-width columns keep the cards the same size
    m1.metric("Wildfire events reported (NASA EONET)", f"{events:,}", border=True)
    m2.metric("Map areas with activity", f"{len({(r['lat'], r['lon']) for r in hist}):,}", border=True)
    m3.metric("Years covered", f"{years[0]}–{years[-1]}", border=True)

# ---------------------------------------------------------------- the two methods
st.header("Three ways to explore risk")
left, mid, right = st.columns(3)

with left.container(border=True, height="stretch"):
    st.subheader("Satellite image analysis", icon=":material/satellite_alt:")
    st.write("Upload a satellite image and a deep-learning classifier tells you whether it shows a wildfire.")
    st.markdown(
        "- ResNet-18 image classifier\n"
        "- Label (wildfire or not) with a confidence score\n"
        "- Grad-CAM heatmap of the areas that influenced the result"
    )

with mid.container(border=True, height="stretch"):
    st.subheader("Risk conditions form", icon=":material/thermostat:")
    st.write("Enter whatever measurements you have and the model predicts which disaster is most likely.")
    st.markdown(
        "- Temperature, humidity, rainfall, wind and quake data\n"
        "- Works with missing values, so fill in only what you know\n"
        "- Top 3 disaster types plus the features that mattered most"
    )

with right.container(border=True, height="stretch"):
    st.subheader("Historical risk map", icon=":material/public:")
    st.write("Explore five years of wildfire activity across the world on an interactive map.")
    st.markdown(
        "- NASA EONET wildfire events, worldwide\n"
        "- Filter by year\n"
        "- Event counts and top hotspots"
    )

# Buttons live in their own row of equal columns so they line up under the cards.
b1, b2, b3 = st.columns(3)
b1.button("Analyze an image", type="primary", icon=":material/arrow_forward:", width="stretch",
    key="go_satellite", on_click=go, args=("Satellite image",))
b2.button("Enter conditions", type="primary", icon=":material/arrow_forward:", width="stretch",
    key="go_conditions", on_click=go, args=("Conditions form",))
b3.button("Open the map", type="primary", icon=":material/arrow_forward:", width="stretch",
    key="go_history", on_click=go, args=("Historical map",))
