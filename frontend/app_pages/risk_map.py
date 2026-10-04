"""Risk map page: five years of global wildfire history (NASA EONET)."""
import numpy as np
import pandas as pd
import streamlit as st

from common import cached_get

st.title("Global wildfire history", icon=":material/public:")

rows = cached_get("/api/regions/history")
if not rows:
    st.warning("No history data. Start the backend, or run `uv run python -m backend.ml.download_fire_history`.",
               icon=":material/info:")
    st.stop()  # nothing below this line runs

df = pd.DataFrame(rows)
years = sorted(df.year.unique())

# A segmented control is a good fit for a handful of mutually exclusive options.
choice = st.segmented_control("Year", ["All years", *map(str, years)], default="All years",
                              required=True, label_visibility="collapsed")
view = df if choice == "All years" else df[df.year == int(choice)]

# Combine the yearly rows into one row per map cell
cells = view.groupby(["lat", "lon"], as_index=False).agg(events=("events", "sum"), hectares=("hectares", "sum"))
cells["radius"] = 40_000 * np.sqrt(cells.events)  # circle size (meters) grows with the number of fires

with st.container(horizontal=True):
    st.metric("Fire events", f"{cells.events.sum():,}", border=True)
    st.metric("Area burned (reported)", f"{cells.hectares.sum() / 1e6:.2f} M ha", border=True)
    st.metric("Map cells affected", f"{len(cells):,}", border=True)

st.map(cells, latitude="lat", longitude="lon", size="radius", color="#d9480f88",
       alt="World map of wildfire events; larger circles mean more events in that area")

left, right = st.columns(2)
with left, st.container(border=True):
    st.subheader("Events per year")
    st.bar_chart(df.groupby("year").events.sum(), alt="Bar chart of wildfire events per year")
with right, st.container(border=True):
    st.subheader("Top hotspots")
    st.dataframe(cells.nlargest(10, "events")[["lat", "lon", "events", "hectares"]], hide_index=True,
                 alt="Table of the ten grid cells with the most wildfire events")

st.caption("Source: NASA EONET wildfire events, last five years, grouped into 2° map cells. "
           "Coverage is much denser from 2024 onward, so earlier years undercount real activity.")
