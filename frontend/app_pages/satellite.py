"""Satellite image page: wildfire classification of an uploaded image."""
import streamlit as st

from common import api

ss = st.session_state

MAX_UPLOAD_BYTES = 4 * 1024 * 1024  # keep in sync with server.maxUploadSize in .streamlit/config.toml

st.title("Satellite image analysis", icon=":material/satellite_alt:")
st.caption("We use deep learning to predict whether an area is at risk of a wildfire. Upload a satellite image to try it.")

left, right = st.columns(2)  # two equal-width columns

with left.container(border=True, height="stretch"):
    st.subheader("Upload")
    up = st.file_uploader("Satellite image", type=["png", "jpg", "jpeg"], label_visibility="collapsed",
                       help="PNG or JPG, up to 4 MB.")
    if up and up.size > MAX_UPLOAD_BYTES:
        st.error(f"{up.name} is {up.size / 1024 / 1024:.1f} MB. Images must be 4 MB or smaller.", icon=":material/error:")
        up = None
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
        st.caption(
            f"The model is {r['confidence']:.0%} confident this image "
            f"{'shows' if is_fire else 'does not show'} a wildfire. "
            "It looks at the whole image, so it does not locate the fire within it."
        )
