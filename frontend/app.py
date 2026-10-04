"""FireWatch frontend.

Streamlit re-runs this whole script top to bottom every time the user interacts
with a widget. Three ideas make that workable:
  1. st.session_state  - remembers values between reruns (e.g. prediction results)
  2. st.cache_data     - skips repeated slow work (e.g. fetching the same data)
  3. layout containers - sidebar, tabs, columns, bordered containers
Look for the numbered comments below.
"""

import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("FIREWATCH_API", "http://localhost:8000")


# ---------------------------------------------------------------- API helpers
def api(method, path, **kw):
    """Call the backend. Shows a friendly message in the UI on failure."""
    try:
        r = requests.request(method, f"{API}{path}", timeout=30, **kw)
        if r.status_code == 422 and isinstance(r.json().get("detail"), str):
            st.warning(r.json()["detail"], icon=":material/info:")
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}", icon=":material/error:")


# 2. Caching: GET data rarely changes, so reuse it for 60s instead of refetching on every rerun.
@st.cache_data(ttl=60, show_spinner=False)
def cached_get(path):
    try:
        r = requests.get(f"{API}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ---------------------------------------------------------------- page setup
# Must be the first Streamlit call. "wide" uses the full browser width.
st.set_page_config(page_title="FireWatch", page_icon=":material/local_fire_department:", layout="wide")

# 1. Session state: a dict that survives reruns. Initialise defaults once.
ss = st.session_state
for k in ("image_result", "disaster_result", "prep"):
    ss.setdefault(k, None)
ss.setdefault("region", "California")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("FireWatch", icon=":material/local_fire_department:")
    st.caption("Wildfire & disaster risk assistant")

    # key="region" links this widget directly to ss.region
    st.text_input("Region", key="region", icon=":material/location_on:")

    online = cached_get("/api/model-info") is not None
    if online:
        st.badge("API connected", icon=":material/check_circle:", color="green")
    else:
        st.badge("API offline", icon=":material/error:", color="red")

    # st.context.theme.type is "light" or "dark". Both palettes are defined in
    # .streamlit/config.toml; users switch via the app menu (top right) > Settings > Theme.
    mode = st.context.theme.type or "light"  # None until the browser reports its theme
    st.caption(f":material/{'dark_mode' if mode == 'dark' else 'light_mode'}: {mode.capitalize()} mode (change in Settings)")

    st.caption(
        "Predictions are informational only and not a substitute for official emergency alerts."
    )

# ---------------------------------------------------------------- header
st.title("Wildfire risk dashboard", icon=":material/dashboard:")
st.caption(f"Analyzing **{ss.region}**. Work through the tabs from left to right.")

# Summary metrics row: shows the latest results at a glance.
img, dis = ss.image_result, ss.disaster_result
with st.container(horizontal=True):
    st.metric("Satellite scan", img["label"] if img else "—", f"{img['confidence']:.0%} confidence" if img else None,
              delta_color="off", border=True)
    st.metric("Likely disaster", dis["top3"][0]["label"] if dis else "—",
              f"{dis['top3'][0]['probability']:.0%} probability" if dis else None, delta_color="off", border=True)
    st.metric("Plan", "Ready" if ss.prep else "Not generated", border=True)

# 3. Tabs split the page into sections. Code in each `with` block renders in that tab.
tab_img, tab_cond, tab_map, tab_plan, tab_model = st.tabs([
    ":material/satellite_alt: Satellite image",
    ":material/thermostat: Conditions",
    ":material/map: Risk map",
    ":material/checklist: Preparedness",
    ":material/monitoring: Model performance",
])

# ---------------------------------------------------------------- tab: image
with tab_img:
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

# ---------------------------------------------------------------- tab: conditions
with tab_cond:
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

# ---------------------------------------------------------------- tab: map
with tab_map:
    rows = cached_get("/api/regions/risk")
    if rows:
        st.map(pd.DataFrame(rows), latitude="lat", longitude="lon", alt="Map of wildfire risk by region")
        with st.expander("Show data", icon=":material/table:"):
            st.dataframe(pd.DataFrame(rows), alt="Table of wildfire risk by region")
    else:
        st.caption("No regional risk data available.")

# ---------------------------------------------------------------- tab: plan
with tab_plan:
    ready = bool(ss.image_result or ss.disaster_result)
    if st.button("Generate plan", type="primary", disabled=not ready, icon=":material/checklist:"):
        ss.prep = api("POST", "/api/preparedness", json={
            "image_result": ss.image_result, "disaster_result": ss.disaster_result, "region": ss.region})
    if not ready:
        st.caption("Run an image or conditions prediction first.")
    if ss.prep:
        st.info(ss.prep["explanation"], icon=":material/lightbulb:")
        # One bordered card per checklist section
        for sec in ss.prep["checklist"]:
            with st.container(border=True):
                st.subheader(sec["title"])
                for i, item in enumerate(sec["items"]):
                    st.checkbox(item, key=f"{sec['title']}-{i}")  # unique key per widget

# ---------------------------------------------------------------- tab: model
with tab_model:
    info = cached_get("/api/model-info") or {}
    for key, title in (("vision", "Satellite image model"), ("disaster", "Disaster type model")):
        m = info.get(key)
        with st.container(border=True):
            st.subheader(title)
            if not m:
                st.caption("No metrics available yet.")
                continue
            st.metric("Accuracy", f"{m['accuracy']:.1%}", f"{m.get('n') or m.get('n_test')} held-out samples",
                      delta_color="off")
            t1, t2 = st.tabs(["Per-class", "Confusion matrix"])
            t1.dataframe(pd.DataFrame(m["per_class"]).T, alt=f"Per-class precision, recall and F1 for the {title.lower()}")
            t2.dataframe(pd.DataFrame(m["confusion_matrix"], index=m["labels"], columns=m["labels"]),
                         alt=f"Confusion matrix for the {title.lower()}")
