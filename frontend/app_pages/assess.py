"""Assess page: the two risk-assessment methods (image and conditions form)."""
import pandas as pd
import streamlit as st

from common import api

ss = st.session_state

st.title("Assess risk", icon=":material/query_stats:")
st.caption(f"Analyzing **{ss.region}**. Pick a method below.")

# Summary metrics row: shows the latest results at a glance.
img, dis = ss.image_result, ss.disaster_result
with st.container(horizontal=True):
    st.metric("Satellite scan", img["label"] if img else "—", f"{img['confidence']:.0%} confidence" if img else None,
              delta_color="off", border=True)
    st.metric("Likely disaster", dis["top3"][0]["label"] if dis else "—",
              f"{dis['top3'][0]['probability']:.0%} probability" if dis else None, delta_color="off", border=True)

# key="method" ties the control to ss.method, which the Home page can set before jumping here.
ss.setdefault("method", "Satellite image")
st.segmented_control("Method", ["Satellite image", "Conditions"], key="method", required=True,
                     label_visibility="collapsed")

# ---------------------------------------------------------------- method 1: image
if ss.method == "Satellite image":
    left, right = st.columns(2)  # two equal-width columns

    with left, st.container(border=True):
        st.subheader("Upload")
        up = st.file_uploader("Satellite image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if up:
            st.image(up, caption=up.name, alt="Uploaded satellite image")
        # Buttons return True only on the rerun right after they are clicked.
        if st.button("Analyze image", type="primary", icon=":material/satellite_alt:", disabled=not up):
            with st.spinner("Analyzing..."):
                ss.image_result = api("POST", "/api/predict/image", files={"file": (up.name, up.getvalue(), up.type)})
            st.rerun()  # redraw so the metrics row above shows the new result

    with right, st.container(border=True):
        st.subheader("Result")
        if not ss.image_result:
            st.caption("Upload an image and click **Analyze image**.")
        else:
            r = ss.image_result
            is_fire = r["label"] == "Wildfire"
            risk = r["confidence"] if is_fire else 1 - r["confidence"]
            if r.get("mock"):
                st.badge("Demo output", icon=":material/science:", color="orange")
            st.badge(r["label"], icon=":material/local_fire_department:" if is_fire else ":material/check:",
                     color="red" if is_fire else "green")
            st.progress(risk, text=f"Wildfire risk: {risk:.0%}")
            st.image(
                f"data:image/png;base64,{r['heatmap_png_b64']}",
                caption="Grad-CAM: regions driving the prediction",
                alt="Satellite image with a heatmap overlay highlighting areas that influenced the prediction",
            )

# ---------------------------------------------------------------- method 2: conditions
else:
    left, right = st.columns(2)

    with left, st.container(border=True):
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
            payload = {"region": ss.region, "description": desc, "temperature_c": temp, "humidity_pct": hum,
                       "rainfall_mm": rain, "wind_speed_kmh": wind, "magnitude": mag, "depth_km": depth}
            with st.spinner("Predicting..."):
                ss.disaster_result = api("POST", "/api/predict/disaster", json=payload)
            st.rerun()

    with right, st.container(border=True):
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
