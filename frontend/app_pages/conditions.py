"""Conditions page: predict the likely disaster type from measurements."""
import pandas as pd
import streamlit as st

from common import api

ss = st.session_state

st.title("Risk conditions form", icon=":material/thermostat:")
st.caption("Enter the measurements you have and the model predicts the most likely disaster.")

left, right = st.columns(2)

with left.container(border=True, height="stretch"):
    st.subheader("Measurements")
    st.caption("Fill in only what you have. Leave the rest blank.")
    # A form batches inputs: the script only reruns when the submit button is pressed.
    with st.form("conditions", border=False):
        c1, c2 = st.columns(2)
        temp = c1.number_input("Temperature (°C)", -60.0, 70.0, None, placeholder="Not measured")
        hum = c2.number_input("Humidity (%)", 0.0, 100.0, None, placeholder="Not measured")
        rain = c1.number_input("Rainfall (mm)", 0.0, 2000.0, None, placeholder="Not measured")
        wind = c2.number_input("Wind speed (km/h)", 0.0, 500.0, None, placeholder="Not measured")
        mag = c1.number_input("Quake magnitude", 0.0, 10.0, None, placeholder="Not measured")
        depth = c2.number_input("Quake depth (km)", 0.0, 800.0, None, placeholder="Not measured")
        desc = st.text_area("Description (optional)")
        submitted = st.form_submit_button("Predict disaster", type="primary", icon=":material/query_stats:")
    if submitted:
        payload = {"description": desc, "temperature_c": temp, "humidity_pct": hum,
                   "rainfall_mm": rain, "wind_speed_kmh": wind, "magnitude": mag, "depth_km": depth}
        with st.spinner("Predicting..."):
            ss.disaster_result = api("POST", "/api/predict/disaster", json=payload)
    
with right.container(border=True, height="stretch"):
    st.subheader("Prediction")
    if not ss.disaster_result:
        st.caption("Submit measurements to see predictions.")
    else:
        d = ss.disaster_result
        if d.get("mock"):
            st.badge("Demo output", icon=":material/science:", color="orange")
        for p in d["top3"]:
            st.progress(p["probability"], text=f"{p['label']}: {p['probability']:.0%}")
        st.markdown("**Key contributing features**")
        st.bar_chart(
            pd.DataFrame(d["top_features"]).set_index("feature"),
            alt="Bar chart of how much each input feature contributed to the prediction",
        )
