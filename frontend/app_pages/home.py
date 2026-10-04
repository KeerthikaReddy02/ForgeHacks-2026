"""Home page: showcases the app and its two ways to assess risk."""
import streamlit as st

from common import cached_get

ss = st.session_state


def start(method):
    """Button callback: remember which method to open, then jump to the Assess page."""
    ss.method = method
    st.switch_page(ss.assess_page)


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
    hectares = sum(r["hectares"] for r in hist)
    years = sorted({r["year"] for r in hist})
    with st.container(horizontal=True):
        st.metric("Wildfire events tracked", f"{events:,}", border=True)
        st.metric("Area burned (reported)", f"{hectares / 1e6:.1f} M ha", border=True)
        st.metric("Years covered", f"{years[0]}–{years[-1]}", border=True)

# ---------------------------------------------------------------- the two methods
st.header("Two ways to assess risk")
left, right = st.columns(2)

with left, st.container(border=True):
    st.subheader("Satellite image analysis", icon=":material/satellite_alt:")
    st.write("Upload a satellite image and a deep-learning classifier tells you whether it shows wildfire.")
    st.markdown(
        "- EfficientNet image classifier\n"
        "- Grad-CAM heatmap highlights the areas that drove the result\n"
        "- Wildfire risk score from 0 to 100%"
    )
    st.button("Analyze an image", type="primary", icon=":material/arrow_forward:",
              key="start_image", on_click=start, args=("Satellite image",))

with right, st.container(border=True):
    st.subheader("Risk conditions form", icon=":material/thermostat:")
    st.write("Enter whatever measurements you have and the model predicts which disaster is most likely.")
    st.markdown(
        "- Temperature, humidity, rainfall, wind and quake data\n"
        "- Works with missing values, so fill in only what you know\n"
        "- Top 3 disaster types plus the features that mattered most"
    )
    st.button("Enter conditions", type="primary", icon=":material/arrow_forward:",
              key="start_conditions", on_click=start, args=("Conditions",))

# ---------------------------------------------------------------- global map teaser
with st.container(border=True):
    st.subheader("Global wildfire history", icon=":material/public:")
    st.write("Explore five years of wildfire activity across the world on an interactive map.")
    st.page_link("app_pages/risk_map.py", label="Open the risk map", icon=":material/map:")
