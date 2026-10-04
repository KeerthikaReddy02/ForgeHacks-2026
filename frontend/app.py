import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("FIREWATCH_API", "http://localhost:8000")


def api(method, path, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=30, **kw)
        if r.status_code == 422 and isinstance(r.json().get("detail"), str):
            st.warning(r.json()["detail"], icon=":material/info:")
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")


st.set_page_config(page_title="FireWatch", page_icon=":material/local_fire_department:")

ss = st.session_state
for k in ("image_result", "disaster_result", "prep"):
    ss.setdefault(k, None)
ss.setdefault("region", "California")

st.title(":material/local_fire_department: FireWatch")
st.warning(
    "Predictions are informational only, not a substitute for official emergency alerts.",
    icon=":material/warning:",
)

with st.container(border=True):
    st.header("Satellite image")
    up = st.file_uploader("Upload a satellite image", type=["png", "jpg", "jpeg"])
    if up and st.button("Analyze image", type="primary", icon=":material/satellite_alt:"):
        ss.image_result = api("POST", "/api/predict/image", files={"file": (up.name, up.getvalue(), up.type)})
    if ss.image_result:
        r = ss.image_result
        risk = r["confidence"] if r["label"] == "Wildfire" else 1 - r["confidence"]
        if r.get("mock"):
            st.caption(":material/science: Demo output (no trained model loaded)")
        st.metric("Result", r["label"], f"{r['confidence']:.0%} confidence", delta_color="off")
        st.progress(risk, text=f"Wildfire risk: {risk:.0%}")
        st.image(
            f"data:image/png;base64,{r['heatmap_png_b64']}",
            caption="Grad-CAM: regions driving the prediction",
            alt="Satellite image with a heatmap overlay highlighting areas that influenced the prediction",
        )

with st.container(border=True):
    st.header("Conditions")
    st.caption("Fill in only the measurements you actually have. Leave the rest blank.")
    with st.form("conditions", border=False):
        region = st.text_input("Region", ss.region)
        with st.container(horizontal=True):
            temp = st.number_input("Temperature (°C)", -60.0, 70.0, None, placeholder="Not measured")
            hum = st.number_input("Humidity (%)", 0.0, 100.0, None, placeholder="Not measured")
            rain = st.number_input("Rainfall (mm)", 0.0, 2000.0, None, placeholder="Not measured")
        with st.container(horizontal=True):
            wind = st.number_input("Wind speed (km/h)", 0.0, 500.0, None, placeholder="Not measured")
            mag = st.number_input("Quake magnitude", 0.0, 10.0, None, placeholder="Not measured")
            depth = st.number_input("Quake depth (km)", 0.0, 800.0, None, placeholder="Not measured")
        desc = st.text_area("Description (optional)")
        submitted = st.form_submit_button("Predict disaster", type="primary", icon=":material/query_stats:")
    if submitted:
        ss.region = region
        payload = {"region": region, "description": desc, "temperature_c": temp, "humidity_pct": hum,
                   "rainfall_mm": rain, "wind_speed_kmh": wind, "magnitude": mag, "depth_km": depth}
        ss.disaster_result = api("POST", "/api/predict/disaster", json=payload)
    if ss.disaster_result:
        d = ss.disaster_result
        if d.get("mock"):
            st.caption(":material/science: Demo output (no trained model loaded)")
        st.subheader("Top 3 predictions")
        for p in d["top3"]:
            st.progress(p["probability"], text=f"{p['label']}: {p['probability']:.0%}")
        st.subheader("Key contributing features")
        st.bar_chart(
            pd.DataFrame(d["top_features"]).set_index("feature"),
            alt="Bar chart of how much each input feature contributed to the prediction",
        )

with st.expander("Model performance", icon=":material/monitoring:"):
    info = api("GET", "/api/model-info") or {}
    for key, title in (("vision", "Satellite image model"), ("disaster", "Disaster type model")):
        m = info.get(key)
        st.markdown(f"**{title}**")
        if not m:
            st.caption("No metrics available yet.")
            continue
        st.write(f"Accuracy: {m['accuracy']:.1%} on {m.get('n') or m.get('n_test')} held-out samples")
        st.dataframe(pd.DataFrame(m["per_class"]).T, alt=f"Per-class precision, recall and F1 for the {title.lower()}")
        st.dataframe(pd.DataFrame(m["confusion_matrix"], index=m["labels"], columns=m["labels"]),
                     alt=f"Confusion matrix for the {title.lower()}")

with st.container(border=True):
    st.header("Risk map")
    rows = api("GET", "/api/regions/risk")
    if rows:
        st.map(pd.DataFrame(rows), latitude="lat", longitude="lon", alt="Map of wildfire risk by region")

with st.container(border=True):
    st.header("Preparedness plan")
    ready = bool(ss.image_result or ss.disaster_result)
    if st.button("Generate plan", disabled=not ready, icon=":material/checklist:"):
        ss.prep = api("POST", "/api/preparedness", json={
            "image_result": ss.image_result, "disaster_result": ss.disaster_result, "region": ss.region})
    if not ready:
        st.caption("Run an image or conditions prediction first.")
    if ss.prep:
        st.info(ss.prep["explanation"])
        for sec in ss.prep["checklist"]:
            st.subheader(sec["title"])
            for i, item in enumerate(sec["items"]):
                st.checkbox(item, key=f"{sec['title']}-{i}")
