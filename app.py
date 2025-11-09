import streamlit as st
import pandas as pd
import joblib

model = joblib.load('airline_model.pkl')

st.set_page_config(page_title="Airline Passenger Satisfaction", layout="wide")
st.title('✈️ Airline Passenger Satisfaction Predictor')
st.write("This app predicts whether a passenger will be satisfied or dissatisfied based on their flight experience.")

st.sidebar.header('Passenger Details')

col1, col2 = st.sidebar.columns(2)

with col1:
    customer_type = st.selectbox('Customer Type', ('Loyal Customer', 'disloyal Customer'))
    type_of_travel = st.selectbox('Type of Travel', ('Business travel', 'Personal Travel'))
    gender = st.selectbox('Gender', ('Female', 'Male'))

with col2:
    age = st.slider('Age', 7, 85, 40)
    flight_distance = st.number_input('Flight Distance (miles)', 31, 5000, 1200)
    customer_class = st.selectbox('Class', ('Business', 'Eco', 'Eco Plus'))

st.subheader('Rate Your Flight Experience (0=Very Poor, 5=Excellent)')

rate_col1, rate_col2, rate_col3 = st.columns(3)

with rate_col1:
    inflight_wifi_service = st.slider('Inflight Wifi Service', 0, 5, 3)
    departure_arrival_time_convenient = st.slider('Departure/Arrival Time Convenient', 0, 5, 3)
    ease_of_online_booking = st.slider('Ease of Online Booking', 0, 5, 3)
    gate_location = st.slider('Gate Location', 0, 5, 3)
    food_and_drink = st.slider('Food and Drink', 0, 5, 3)
    online_boarding = st.slider('Online Boarding', 0, 5, 3)

with rate_col2:
    seat_comfort = st.slider('Seat Comfort', 0, 5, 3)
    inflight_entertainment = st.slider('Inflight Entertainment', 0, 5, 3)
    on_board_service = st.slider('On-board Service', 0, 5, 3)
    leg_room_service = st.slider('Leg Room Service', 0, 5, 3)
    baggage_handling = st.slider('Baggage Handling', 0, 5, 3)
    checkin_service = st.slider('Check-in Service', 0, 5, 3)

with rate_col3:
    inflight_service = st.slider('Inflight Service', 0, 5, 3)
    cleanliness = st.slider('Cleanliness', 0, 5, 3)
    departure_delay_in_minutes = st.number_input('Departure Delay (minutes)', 0, 2000, 0)

if st.button('Predict Satisfaction', use_container_width=True):

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
        'Departure Delay in Minutes': departure_delay_in_minutes
    }

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    prob_dissatisfied = prediction_proba[0][0]
    prob_satisfied = prediction_proba[0][1]

    if prediction[0] == 1:
        st.success(f"**Prediction: Satisfied** (Probability: {prob_satisfied:.1%})")
        st.balloons()
    else:
        st.error(f"**Prediction: Neutral or Dissatisfied** (Probability: {prob_dissatisfied:.1%})")

    st.subheader("Prediction Probabilities:")
    st.write(f"Probability of being 'Neutral or Dissatisfied': {prob_dissatisfied:.1%}")
    st.write(f"Probability of being 'Satisfied': {prob_satisfied:.1%}")