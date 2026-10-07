"""FireWatch frontend: the entry point.

Streamlit re-runs a script top to bottom every time the user interacts with a widget.
Ideas to look for while reading:
  1. st.session_state  - remembers values between reruns (shared by all pages)
  2. st.cache_data     - skips repeated work (see common.py)
  3. st.navigation     - turns files in app_pages/ into a multipage app
  4. layout containers - sidebar, tabs, columns, bordered containers
  5. st.dialog         - a pop-up window

This file runs on EVERY page: it sets up the shared bits (sidebar, footer), then
runs whichever page is selected.
"""

import pandas as pd
import streamlit as st

from common import cached_get

# Must be the first Streamlit call. "wide" uses the full browser width.
st.set_page_config(page_title="FireWatch", page_icon=":material/local_fire_department:", layout="wide")

# 1. Session state: a dict that survives reruns. Initialise defaults once.
ss = st.session_state
for k in ("image_result", "disaster_result"):
    ss.setdefault(k, None)

# 3. Navigation: each st.Page points to a script file. The first page is the default.
pages = [
    st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("app_pages/satellite.py", title="Satellite image", icon=":material/satellite_alt:"),
    st.Page("app_pages/conditions.py", title="Conditions form", icon=":material/thermostat:"),
    st.Page("app_pages/history.py", title="Historical map", icon=":material/public:"),
]
page = st.navigation(pages)
ss.pages = {p.title: p for p in pages}  # lets other pages jump to a page with st.switch_page

# ---------------------------------------------------------------- model performance dialog
# 5. @st.dialog turns a function into a pop-up. Calling it opens the modal.
@st.dialog("Model performance", width="large")
def show_model_performance():
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


# ---------------------------------------------------------------- sidebar (shown on every page)
with st.sidebar:
    st.caption("Wildfire & disaster risk assistant")

    if cached_get("/api/model-info") is not None:
        st.badge("API connected", icon=":material/check_circle:", color="green")
    else:
        st.badge("API offline", icon=":material/error:", color="red")

    # st.context.theme.type is "light" or "dark". Both palettes are defined in
    # .streamlit/config.toml; users switch via the app menu (top right) > Settings > Theme.
    mode = st.context.theme.type or "light"  # None until the browser reports its theme
    st.caption(f":material/{'dark_mode' if mode == 'dark' else 'light_mode'}: {mode.capitalize()} mode (change in Settings)")

    st.caption("Predictions are informational only and not a substitute for official emergency alerts.")

    # Sidebar content renders below the page links. The key gives this container a CSS class
    # (.st-key-nav_footer) that the stylesheet below uses to pin it to the bottom.
    with st.container(key="nav_footer"):
        if st.button("Model performance", icon=":material/monitoring:", width="stretch"):
            show_model_performance()

# Streamlit has no native "sidebar footer", so this one small CSS rule pushes the container to the bottom.
st.html("""<style>
[data-testid="stSidebarContent"] { position: relative; }
.st-key-nav_footer { position: absolute; bottom: 1rem; left: 1rem; width: calc(100% - 2rem); }
</style>""")

# ---------------------------------------------------------------- run the selected page
page.run()
