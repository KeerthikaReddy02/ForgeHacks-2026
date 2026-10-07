"""Satellite image page: wildfire classification of an uploaded image."""
import streamlit as st

from common import api

ss = st.session_state

st.title("Satellite image analysis", icon=":material/satellite_alt:")
st.caption("Upload a satellite image to check whether it shows a wildfire.")

left, right = st.columns(2)  # two equal-width columns

with left.container(border=True, height="stretch"):
    st.subheader("Upload")
    up = st.file_uploader("Satellite image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if up:
        st.image(up, caption=up.name, alt="Uploaded satellite image")
    # Buttons return True only on the rerun right after they are clicked.
    if st.button("Analyze image", type="primary", icon=":material/satellite_alt:", disabled=not up):
        with st.spinner("Analyzing..."):
            ss.image_result = api("POST", "/api/predict/image", files={"file": (up.name, up.getvalue(), up.type)})

with right.container(border=True, height="stretch"):
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
