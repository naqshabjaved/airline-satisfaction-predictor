import streamlit as st
import pandas as pd
import joblib
import numpy as np

try:
    model = joblib.load('airline_model.pkl')
except FileNotFoundError:
    st.error("Error: 'airline_model.pkl' not found. Please run your notebook to create it.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Airline Passenger Satisfaction", layout="wide")
st.title('Airline Passenger Satisfaction Predictor')
st.write("""
This app predicts whether a passenger will be **Satisfied** or **Neutral/Dissatisfied** based on their flight experience. 
The model is a Random Forest Classifier trained on airline survey data.
""")
st.markdown("---")

col1, col2 = st.columns(2)

# --- Column 1: Passenger Details ---
with col1:
    st.header("Passenger Details")
    age = st.slider('Age', 7, 85, 40)
    customer_type = st.selectbox('Customer Type', ('Loyal Customer', 'disloyal Customer'))
    type_of_travel = st.selectbox('Type of Travel', ('Business travel', 'Personal Travel'))
    
with col2:
    st.header("Flight Details")
    customer_class = st.selectbox('Class', ('Business', 'Eco', 'Eco Plus'))
    flight_distance = st.number_input('Flight Distance (miles)', min_value=30, max_value=5000, value=1200)
    departure_delay_in_minutes = st.number_input('Departure Delay (minutes)', min_value=0, max_value=2000, value=0)

st.markdown("---")

st.header("Service Ratings (0 = Not Used, 1 = Very Poor, 5 = Excellent)")

rate_col1, rate_col2, rate_col3 = st.columns(3)

with rate_col1:
    inflight_wifi_service = st.slider('Inflight Wifi Service', 0, 5, 3)
    departure_arrival_time_convenient = st.slider('Departure/Arrival Time Convenient', 0, 5, 3)
    ease_of_online_booking = st.slider('Ease of Online Booking', 0, 5, 3)
    gate_location = st.slider('Gate Location', 0, 5, 3)
    food_and_drink = st.slider('Food and Drink', 0, 5, 3)

with rate_col2:
    online_boarding = st.slider('Online Boarding', 0, 5, 3)
    seat_comfort = st.slider('Seat Comfort', 0, 5, 3)
    inflight_entertainment = st.slider('Inflight Entertainment', 0, 5, 3)
    on_board_service = st.slider('On-board Service', 0, 5, 3)
    leg_room_service = st.slider('Leg Room Service', 0, 5, 3)

with rate_col3:
    baggage_handling = st.slider('Baggage Handling', 0, 5, 3)
    checkin_service = st.slider('Checkin Service', 0, 5, 3)
    inflight_service = st.slider('Inflight Service', 0, 5, 3)
    cleanliness = st.slider('Cleanliness', 0, 5, 3)
    
    gender = 'Female' 

st.markdown("---")

if st.button('Predict Satisfaction', use_container_width=True, type="primary"):
    
    input_data = {
        'Gender': gender,
        'Customer Type': customer_type,
        'Age': age,
        'Type of Travel': type_of_travel,
        'Class': customer_class,
        'Flight Distance': flight_distance,
        'Inflight wifi service': inflight_wifi_service,
        'Departure/Arrival time convenient': departure_arrival_time_convenient,
        'Ease of Online booking': ease_of_online_booking,
        'Gate location': gate_location,
        'Food and drink': food_and_drink,
        'Online boarding': online_boarding,
        'Seat comfort': seat_comfort,
        'Inflight entertainment': inflight_entertainment,
        'On-board service': on_board_service,
        'Leg room service': leg_room_service,
        'Baggage handling': baggage_handling,
        'Checkin service': checkin_service,
        'Inflight service': inflight_service,
        'Cleanliness': cleanliness,
        'Departure Delay in Minutes': departure_delay_in_minutes,
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(f"**Prediction: SATISFIED** (Probability: {prediction_proba[1]:.1%})")
            st.balloons()
        else:
            st.error(f"**Prediction: NEUTRAL / DISSATISFIED** (Probability: {prediction_proba[0]:.1%})")
        
        st.write("This prediction indicates the passenger's likely sentiment based on the provided details.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
