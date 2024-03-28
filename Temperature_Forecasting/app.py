import streamlit as st
import calendar
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import matplotlib.pyplot as plt

# Load the SARIMAX model
with open('fitted_sarimax_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to get average temperature prediction
def get_temperature_prediction(month, year):
    # Make prediction
    prediction = model.get_prediction(start=f'{year}-{month}-01', end=f'{year}-{month}-28', dynamic=False)
    forecast = prediction.predicted_mean.mean() # Change to appropriate function/method depending on your model
    return forecast

# Function to plot temperature vs time graph
def plot_temperature_vs_time(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date)
    predictions = []
    for date in dates:
        prediction = get_temperature_prediction(date.month, date.year)
        predictions.append(prediction)
    plt.figure(figsize=(10, 6))
    plt.plot(dates, predictions, marker='o', linestyle='-')
    plt.title('Temperature Forecast Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Temperature (°C)')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)

# Streamlit app
def main():
    st.title("Bhubaneswar Temperature Forecasting App")
    st.markdown("---")
    st.write("Enter the month and year to get the average monthly temperature prediction.")

    # User input for single month prediction
    col1, col2 = st.columns([2, 3])
    with col1:
        month_names = [calendar.month_name[i] for i in range(1, 13)]
        month = st.selectbox("Select Month", options=month_names)
    with col2:
        year = st.number_input("Enter Year", min_value=2000, max_value=3000, step=1, value=2024)

    if st.button("Get Prediction"):
        month_number = month_names.index(month) + 1
        temperature_prediction = get_temperature_prediction(month_number, year)
        st.markdown("---")
        st.success(f"Average temperature prediction for {year}-{month}: **{temperature_prediction:.2f} °C**")

    st.markdown("---")
    st.write("Select a date range to view temperature vs. time graph.")

    # User input for date range
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    if st.button("Plot Temperature vs. Time"):
        if start_date <= end_date:
            plot_temperature_vs_time(start_date, end_date)
        else:
            st.error("Error: End date must be after start date.")

if __name__ == "__main__":
    main()
