"""Historical risk map: every wildfire event NASA EONET reports for the selected year (live)."""
from datetime import date

import pandas as pd
import pydeck as pdk
import streamlit as st

from common import cached_get

st.title("Historical risk map", icon=":material/public:")

# Step through the last five calendar years with < and >. The chosen year lives in session state.
this_year = date.today().year
first_year = this_year - 4
st.session_state.setdefault("hist_year", this_year)
year = st.session_state.hist_year


def step(delta):
    """Button callback: move to the previous (-1) or next (+1) year."""
    st.session_state.hist_year += delta


prev_col, label_col, next_col = st.columns([1, 3, 1], vertical_alignment="center")
prev_col.button("", icon=":material/chevron_left:", key="prev_year", on_click=step, args=(-1,),
                disabled=year <= first_year, width="stretch", help="Previous year")
label_col.markdown(f"### {year}", text_alignment="center")
next_col.button("", icon=":material/chevron_right:", key="next_year", on_click=step, args=(1,),
                disabled=year >= this_year, width="stretch", help="Next year")

# Fetched live from NASA through our backend. cache_data keeps each year for an hour, so
# switching back to a year you already loaded is instant.
with st.spinner(f"Loading {year} wildfire events from NASA EONET (a busy year can take ~15 seconds)..."):
    rows = cached_get(f"/api/regions/history?year={year}", timeout=120)
if rows is None:
    st.error("Could not load live data from NASA EONET. Is the backend running?", icon=":material/error:")
    st.stop()  # nothing below this line runs
if not rows:
    st.info(f"NASA EONET reports no wildfire events for {year}.", icon=":material/info:")
    st.stop()

df = pd.DataFrame(rows)
df["month"] = pd.to_datetime(df.date).dt.month

m1, m2 = st.columns(2)
m1.metric(f"Wildfire events in {year}", f"{len(df):,}", border=True)
m2.metric("Busiest month", pd.Timestamp(2000, int(df.month.value_counts().idxmax()), 1).strftime("%B"), border=True)

# st.map zooms to fit the data, so we use pydeck to pin the camera to a zoomed-out world view.
st.pydeck_chart(
    pdk.Deck(
        layers=[pdk.Layer("ScatterplotLayer", df, get_position="[lon, lat]", get_radius=20_000,
                          radius_min_pixels=2, get_fill_color=[217, 72, 15, 150])],
        initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=0.6),
        map_style="light",
    ),
    height=500,
)

left, right = st.columns(2)
with left.container(border=True, height="stretch"):
    st.subheader("Events per month")
    st.bar_chart(df.groupby("month").size(), alt=f"Bar chart of wildfire events per month in {year}")
with right.container(border=True, height="stretch"):
    st.subheader("Latest events")
    st.dataframe(df.sort_values("date", ascending=False).head(10)[["date", "title"]], hide_index=True,
                 alt=f"Table of the ten most recent wildfire events in {year}")

st.caption("Source: NASA EONET (live). EONET coverage varies by year, so counts reflect what it reports, "
           "not every fire that occurred.")
