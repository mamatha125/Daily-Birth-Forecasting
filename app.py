import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from prophet import Prophet

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Page configuration
st.set_page_config(
    page_title="Birth Forecasting & Healthcare Planning",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 AI-Based Daily Birth Forecasting and Healthcare Resource Planning")
st.write(
    "This dashboard uses machine learning-based forecasting "
    "to estimate future birth counts and support healthcare resource planning."
)


# Load forecast data
forecast = pd.read_csv(
    "data/final_birth_forecast.csv"
)

forecast["ds"] = pd.to_datetime(forecast["ds"])


# Load evaluation results
evaluation = pd.read_csv(
    "data/evaluation_results.csv"
)


# Load healthcare resource plan
resource_plan = pd.read_csv(
    "data/healthcare_resource_plan.csv"
)

resource_plan["ds"] = pd.to_datetime(
    resource_plan["ds"]
)
st.header("🔮 Generate New Forecast")

forecast_months = st.number_input(
    "Number of months to forecast",
    min_value=1,
    max_value=12,
    value=6,
    step=1
)

if st.button("🚀 Generate Forecast"):

    historical_data = pd.read_csv(
        DATA_DIR / "monthly_births_2024.csv"
    )

    historical_data["month"] = (
        historical_data["month"]
        .astype(str)
        .str.zfill(2)
    )

    historical_data["ds"] = pd.to_datetime(
        "2024-" +
        historical_data["month"] +
        "-01"
    )

    historical_data["y"] = historical_data["births"]

    prophet_data = historical_data[["ds", "y"]]

    new_model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    new_model.fit(prophet_data)

    future = new_model.make_future_dataframe(
        periods=forecast_months,
        freq="MS"
    )

    new_forecast = new_model.predict(future)

    future_predictions = new_forecast[
        new_forecast["ds"] >
        historical_data["ds"].max()
    ].copy()

    future_predictions["Predicted_Births"] = (
        future_predictions["yhat"]
        .round()
        .astype(int)
    )

    future_predictions["Lower_Limit"] = (
        future_predictions["yhat_lower"]
        .round()
        .astype(int)
    )

    future_predictions["Upper_Limit"] = (
        future_predictions["yhat_upper"]
        .round()
        .astype(int)
    )

    st.success("New forecast generated successfully!")

    st.dataframe(
        future_predictions[
            [
                "ds",
                "Predicted_Births",
                "Lower_Limit",
                "Upper_Limit"
            ]
        ],
        use_container_width=True
    )

# -------------------------------
# Forecast Section
# -------------------------------

st.header("📈 Birth Forecast")

st.dataframe(
    forecast,
    use_container_width=True
)


# Forecast chart

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    forecast["ds"],
    forecast["Predicted_Births"],
    marker="o"
)

ax.set_title("Predicted Births")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Births")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)


# -------------------------------
# Evaluation Section
# -------------------------------

st.header("📊 Model Evaluation")

col1, col2, col3 = st.columns(3)

mae = evaluation.loc[
    evaluation["Metric"] == "MAE", "Value"
].iloc[0]

rmse = evaluation.loc[
    evaluation["Metric"] == "RMSE", "Value"
].iloc[0]

mape = evaluation.loc[
    evaluation["Metric"] == "MAPE", "Value"
].iloc[0]

col1.metric("MAE", round(mae, 2))
col2.metric("RMSE", round(rmse, 2))
col3.metric("MAPE", f"{mape:.2f}%")


# -------------------------------
# Healthcare Planning
# -------------------------------

st.header("🏥 Healthcare Resource Planning")

st.dataframe(
    resource_plan[
        [
            "ds",
            "Predicted_Births",
            "Estimated_Beds",
            "Estimated_Nurses"
        ]
    ],
    use_container_width=True
)


# Resource chart

fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(
    resource_plan["ds"],
    resource_plan["Estimated_Beds"],
    marker="o",
    label="Estimated Beds"
)

ax2.plot(
    resource_plan["ds"],
    resource_plan["Estimated_Nurses"],
    marker="o",
    label="Estimated Nurses"
)

ax2.set_title("Healthcare Resource Planning")
ax2.set_xlabel("Date")
ax2.set_ylabel("Estimated Requirement")

ax2.legend()

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig2)


st.success(
    "Forecasting and healthcare resource planning completed successfully!"
)