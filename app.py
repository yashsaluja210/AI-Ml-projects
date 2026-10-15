import joblib
import pandas as pd
import streamlit as st

# 1. Load the saved model and column structure
model = joblib.load("xgb_model.pkl")
model_columns = joblib.load("columns.pkl")

# 2. App Title & Description
st.title("🏨 Hotel Booking Cancellation Predictor")
st.write("Enter the booking details below to predict if the guest will cancel.")

# 3. Collect User Inputs
number_of_adults = st.number_input(
    "Number of Adults", min_value=1, max_value=10, value=2
)
number_of_children = st.number_input(
    "Number of Children", min_value=0, max_value=10, value=0
)
number_of_weekend_nights = st.number_input(
    "Weekend Nights", min_value=0, max_value=10, value=1
)
number_of_week_nights = st.number_input(
    "Week Nights", min_value=0, max_value=30, value=2
)

type_of_meal = st.selectbox(
    "Type of Meal Plan",
    ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"],
)
car_parking_space = st.selectbox("Car Parking Space", [0, 1])
room_type = st.selectbox(
    "Room Type",
    [
        "Room_Type 1",
        "Room_Type 2",
        "Room_Type 3",
        "Room_Type 4",
        "Room_Type 5",
        "Room_Type 6",
        "Room_Type 7",
    ],
)
lead_time = st.number_input("Lead Time", min_value=0, max_value=500, value=30)
market_segment_type = st.selectbox(
    "Market Segment Type",
    ["Online", "Offline", "Corporate", "Complementary", "Aviation"],
)

repeated = st.selectbox("Repeated Guest", [0, 1])
p_c = st.number_input("Previous Cancellations (P-C)", min_value=0, value=0)
p_not_c = st.number_input(
    "Previous Bookings Not Canceled (P-not-C)", min_value=0, value=0
)
average_price = st.number_input(
    "Average Price", min_value=0.0, max_value=1000.0, value=100.0
)
special_requests = st.selectbox("Special Requests", [0, 1, 2, 3, 4, 5])
date_of_reservation = st.date_input("Date of Reservation")

# 4. Predict Button
if st.button("Predict"):
    # Extract date parts
    res_year = date_of_reservation.year
    res_month = date_of_reservation.month
    res_week = date_of_reservation.weekday()

    # Feature Engineering (same as training notebook)
    total_nights = number_of_weekend_nights + number_of_week_nights
    total_family_member = number_of_adults + number_of_children

    # Build input dictionary
    input_data = {
        "number of adults": number_of_adults,
        "number of children": number_of_children,
        "number of weekend nights": number_of_weekend_nights,
        "number of week nights": number_of_week_nights,
        "type of meal": type_of_meal,
        "car parking space": car_parking_space,
        "room type": room_type,
        "lead time": lead_time,
        "market segment type": market_segment_type,
        "repeated": repeated,
        "P-C": p_c,
        "P-not-C": p_not_c,
        "average price": average_price,
        "special requests": special_requests,
        "reservation_year": float(res_year),
        "reservation_month": float(res_month),
        "reservation_week": float(res_week),
        "total_nights": total_nights,
        "total_family_member": total_family_member,
    }

    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])

    # One-hot encoding matching your training pipeline
    df_input = pd.get_dummies(
        df_input,
        columns=["type of meal", "room type", "market segment type"],
        drop_first=True,
        dtype=int,
    )

    # Reindex to ensure exact match with training columns (fill missing dummy columns with 0)
    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    # Make Prediction
    prediction = model.predict(df_input)[0]

    # Display Result
    if prediction == 1:
        st.error("Prediction: The booking is likely to be **Canceled** ❌")
    else:
        st.success("Prediction: The booking is likely **Not Canceled** ✅")