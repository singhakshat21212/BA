import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.sav')
scaler = joblib.load('scaler.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the features to predict if a delivery will be delayed.')

# Create input widgets for each feature
# Using some reasonable ranges and default values for demonstration
delivery_distance = st.slider('Delivery Distance (km)', 0.0, 100.0, 25.0)
traffic_congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-5)', 1, 5, 3)
delivery_slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
driver_experience = st.slider('Driver Experience (years)', 0, 20, 5)
num_stops = st.slider('Number of Stops', 0, 10, 3)
vehicle_age = st.slider('Vehicle Age (years)', 0, 15, 3)
road_condition_score = st.slider('Road Condition Score (1-5)', 1, 5, 3)
package_weight = st.slider('Package Weight (kg)', 0.5, 50.0, 5.0)
fuel_efficiency = st.slider('Fuel Efficiency (km/L)', 5.0, 30.0, 15.0)
warehouse_processing_time = st.slider('Warehouse Processing Time (hours)', 0.5, 12.0, 2.0)

# Create a DataFrame from user inputs
input_data = pd.DataFrame([
    {
        'Delivery_Distance': delivery_distance,
        'Traffic_Congestion': traffic_congestion,
        'Weather_Condition': weather_condition,
        'Delivery_Slot': delivery_slot,
        'Driver_Experience': driver_experience,
        'Num_Stops': num_stops,
        'Vehicle_Age': vehicle_age,
        'Road_Condition_Score': road_condition_score,
        'Package_Weight': package_weight,
        'Fuel_Efficiency': fuel_efficiency,
        'Warehouse_Processing_Time': warehouse_processing_time
    }
])

# Scale the input data using the loaded scaler
input_data_scaled = scaler.transform(input_data)

if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    if prediction[0] == 1:
        st.error(f"**Prediction: Delivery will likely be DELAYED!** (Probability: {prediction_proba[0][1]:.2f})")
    else:
        st.success(f"**Prediction: Delivery is expected to be ON TIME.** (Probability: {prediction_proba[0][0]:.2f})")

st.write("Note: A '1' for Traffic Congestion, Weather Condition, Road Condition Score means very good/low, while '5' means very bad/high (e.g., heavy traffic, severe weather).")
