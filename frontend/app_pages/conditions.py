import pandas as pd
import streamlit as st

from common import api


ss = st.session_state


st.title(
    "Wildfire Risk Conditions",
    icon=":material/local_fire_department:",
)

st.caption(
    "Enter location and weather conditions to estimate wildfire risk."
)


left, right = st.columns(2)


# LEFT: INPUT


with left.container(
    border=True,
    height="stretch"
):

    st.subheader("Conditions")

    st.caption("Enter the six measurements used by the wildfire model.")
    with st.form(
        "wildfire_conditions",
        border=False
    ):

        c1, c2 = st.columns(2)

        latitude = c1.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=38.0,
        )

        longitude = c2.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-77.0,
        )

        max_temp = c1.number_input(
            "Maximum Temperature (°C)",
            min_value=-60.0,
            max_value=70.0,
            value=29.0,
        )

        precipitation = c2.number_input(
            "Precipitation (mm)",
            min_value=0.0,
            max_value=2000.0,
            value=12.0,
        )

        wind_speed = c1.number_input(
            "Wind Speed",
            min_value=0.0,
            max_value=500.0,
            value=10.0,
        )

        min_humidity = c2.number_input(
            "Minimum Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
        )

        submitted = st.form_submit_button(
            "Predict Wildfire Risk",
            type="primary",
            icon=":material/query_stats:",
        )


    if submitted:

        payload = {
            "latitude": latitude,
            "longitude": longitude,
            "max_temp": max_temp,
            "precipitation": precipitation,
            "wind_speed": wind_speed,
            "min_humidity": min_humidity,
        }

        with st.spinner(
            "Calculating wildfire risk..."
        ):

            try:

                ss.wildfire_result = api(
                    "POST",
                    "/api/predict/wildfire",
                    json=payload,
                )

                ss.wildfire_error = None

            except Exception as e:

                ss.wildfire_result = None

                ss.wildfire_error = str(e)


# RIGHT: PREDICTION

with right.container(
    border=True,
    height="stretch"
):

    st.subheader("Wildfire Risk Prediction")
    if ss.get("wildfire_error"):

        st.error(f"Prediction failed: {ss.wildfire_error}")

    elif not ss.get("wildfire_result"):
        st.caption("Submit the conditions to see the wildfire prediction.")

    else:

        result = ss.wildfire_result
        probability = result["probability"]
        risk_level = result["risk_level"]
        # Probability
        st.metric("Wildfire Probability",f"{probability:.2%}")

        st.progress(
            probability,
            text=f"Risk level: {risk_level}",
        )

        # Risk level

        if risk_level == "High Risk":

            st.error(
                f"🔥 **{risk_level}**"
            )

        elif risk_level == "Medium Risk":

            st.warning(
                f"⚠️ **{risk_level}**"
            )

        elif risk_level == "Low Risk":

            st.info(
                f"**{risk_level}**"
            )

        elif risk_level == "Very Low Risk":

            st.success(
                f"**{risk_level}**"
            )

        else:

            st.success(
                f"**{risk_level}**"
            )

        # SHAP explanation

        st.markdown(
            "### Key Contributing Factors"
        )

        factors = result.get(
            "top_factors",
            []
        )

        if factors:

            chart_data = pd.DataFrame(
                {
                    factor["feature"]:
                    [factor["shap"]]
                    for factor in factors
                }
            ).T

            chart_data.columns = [
                "SHAP"
            ]

            st.bar_chart(
                chart_data,
                horizontal=True,
            )



            for factor in factors:

                if factor["direction"] == "increases":

                    st.markdown(
                        f"**{factor['description']}** "
                        f"increased wildfire risk."
                    )

                else:

                    st.markdown(
                        f"**{factor['description']}** "
                        f"reduced wildfire risk."
                    )
            st.markdown(
    """
    <div style="
        text-align: justify;
        padding: 12px 16px;
        margin-top: 14px;
        margin-bottom: 10px;
        border-radius: 8px;
        border: 1px solid #B8DDF5;
        background-color: #EAF6FF;
        color: #1F2937;
        line-height: 1.5;
    ">
        <b>How to read the SHAP plot:</b>
        Positive values mean the feature increased the predicted wildfire
        risk. Negative values mean the feature reduced the predicted
        wildfire risk. The farther the value is from zero, the stronger
        its influence on this prediction.
    </div>
    """,
    unsafe_allow_html=True
)
            
#             st.info(
#     "How to read the SHAP plot: Positive values mean the feature "
#     "increased the predicted wildfire risk. Negative values mean the "
#     "feature reduced the predicted wildfire risk. The farther the value "
#     "is from zero, the stronger its influence on this prediction."
# )
            