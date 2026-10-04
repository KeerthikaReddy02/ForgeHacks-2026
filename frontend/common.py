"""Helpers shared by every page."""
import os

import requests
import streamlit as st

API = os.getenv("FIREWATCH_API", "http://localhost:8000")


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


# Caching: GET data rarely changes, so reuse it for 60s instead of refetching on every rerun.
@st.cache_data(ttl=60, show_spinner=False)
def cached_get(path):
    try:
        r = requests.get(f"{API}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None
