import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Inventory Forecasting & Recommendation Dashboard")

# Dummy example data, replace with your actual forecast + decision data
data = {
    "Date": pd.date_range(start="2025-12-01", periods=30),
    "Actual Demand": np.random.poisson(400, 30),
    "Forecast Demand": np.random.poisson(420, 30)
}

df = pd.DataFrame(data)

# Plot Actual vs Forecast
st.subheader("Demand Forecast vs Actual")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Date"], df["Actual Demand"], label="Actual Demand")
ax.plot(df["Date"], df["Forecast Demand"], label="Forecast Demand", linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.legend()
st.pyplot(fig)

# Inventory decision summary (replace with your real decision dict)
inventory_decision = {
    "avg_daily_forecast": 438.62,
    "forecast_std": 316.05,
    "lead_time_days": 3.0,
    "demand_during_lead": 1315.87,
    "safety_stock": 903.22,
    "reorder_point": 2219.09,
    "recommended_order_qty": 3474,
    "days_of_cover": 1.14,
    "priority": "HIGH"
}

st.subheader("Inventory Decision Summary")
for k, v in inventory_decision.items():
    st.write(f"**{k.replace('_', ' ').title()}:** {v}")

# Supplier recommendation example
supplier_info = {
    'best_supplier': 'A',
    'order_qty': 3474,
    'expected_cost': 5960.0,
    'lead_time': 1,
    'reliability': 0.98
}

st.subheader("Supplier Recommendation")
st.write(f"**Best Supplier:** {supplier_info['best_supplier']}")
st.write(f"**Recommended Order Qty:** {supplier_info['order_qty']}")
st.write(f"**Expected Cost (₹):** {supplier_info['expected_cost']}")
st.write(f"**Lead Time (days):** {supplier_info['lead_time']}")
st.write(f"**Reliability:** {supplier_info['reliability']}")

# What-If Simulator (interactive sliders)
st.subheader("What-If Simulator")
demand_increase = st.slider("Increase demand by (%)", 0, 100, 0)
lead_time_change = st.slider("Change lead time (days)", -5, 10, 0)

st.write(f"Simulated demand increase: {demand_increase}%")
st.write(f"Simulated lead time change: {lead_time_change} days")
