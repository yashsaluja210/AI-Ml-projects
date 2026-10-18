import joblib
import streamlit as st
import pandas as pd

# Load the save data
model=joblib.load("best_rf_model.pkl")
model_columns=joblib.load("columns1.pkl")
scaler=joblib.load("scaler1.pkl")

# App title and Description
st.title(" BigMart Sales Prediction App")
st.write("Enter the Following Details to predict the Outlet Sales ")

# Collect User Inputs
item_category=st.selectbox("Type of Item Category",["Food","Drink","Non-Consumable"])
item_type=st.selectbox("Item Type",["Fruits and Vegetables","Snack Foods","Household","Frozen Foods","Dairy","Canned","Baking Goods","Health and Hygiene",       
"Soft Drinks","Meat","Breads","Hard Drinks","Others","Starchy Foods","Breakfast","Seafood"])
item_fat_content=st.selectbox("Fat Content",["Low Fat","Regular"])
item_weight=st.number_input("Item Weight",min_value=0.1,max_value=30.0,value=0.1)
outlet_identifier = st.selectbox(
    "Outlet Identifier",
    [
        "OUT049",
        "OUT018",
        "OUT010",
        "OUT013",
        "OUT027",
        "OUT045",
        "OUT017",
        "OUT046",
        "OUT035",
        "OUT019",
    ]),
outlet_location=st.selectbox("Location of Outlet",["Tier 1","Tier 2","Tier 3"])
outlet_size=st.selectbox("Outlet Size",["Small","Medium","High"])
item_visibility=st.slider("Item Visibility",min_value=0.000001,max_value=0.500000,value=0.000001)
item_mrp=st.number_input("Item MRP",min_value=1,max_value=500,value=10)
outlet_type=st.selectbox("Outlet Type",["Supermarket Type1","Supermarket Type2","Grocery Store","Supermarket Type3"])
outlet_age=st.slider("Outlet TimePeriod",min_value=1,max_value=50,value=20)



# 3. Predict Button
if st.button("Predict"):



  # Feature Engineering (matching training pipeline)
  price_per_weight = item_mrp / item_weight

  # Dataset Mean values computed during training
  item_type_mean_mrp_dict = {
      "Fruits and Vegetables": 144.58,
      "Snack Foods": 146.19,
      "Household": 149.42,
      "Frozen Foods": 138.50,
      "Dairy": 148.50,
      "Canned": 139.79,
      "Baking Goods": 126.30,
      "Health and Hygiene": 130.81,
      "Soft Drinks": 131.49,
      "Meat": 139.88,
      "Breads": 140.95,
      "Hard Drinks": 137.07,
      "Others": 132.85,
      "Starchy Foods": 147.85,
      "Breakfast": 141.78,
      "Seafood": 141.84,
  }
  item_type_mean_mrp = item_type_mean_mrp_dict.get(item_type, 140.99)

  outlet_type_mean_sales_dict = {
      "Supermarket Type1": 2316.18,
      "Supermarket Type2": 1995.49,
      "Grocery Store": 339.83,
      "Supermarket Type3": 3694.03,
  }
  outlet_type_mean_sales = outlet_type_mean_sales_dict.get(
      outlet_type, 2181.28
  )

  # Label encoding mappings
  cat_mapping = {"Drink": 0, "Food": 1, "Non-Consumable": 2}
  item_type_category_encoded = cat_mapping.get(item_category, 1)
  outlet_id_encoded = 9  # default matching OUT049 label index

  # Build input dictionary
  input_data = {
      "Item_Weight": item_weight,
      "Item_Visibility": item_visibility,
      "Item_MRP": item_mrp,
      "Outlet_Identifier": outlet_id_encoded,
      "Item_type_category": item_type_category_encoded,
      "outlet_age": outlet_age,
      "Price_Per_Weight": price_per_weight,
      "Item_Type_Mean_MRP": item_type_mean_mrp,
      "Outlet_Type_Mean_Sales": outlet_type_mean_sales,
      "Item_Fat_Content": item_fat_content,
    "Item_Type": item_type,
    "Outlet_Size": outlet_size,
    "Outlet_Location_Type": outlet_location,
    "Outlet_Type": outlet_type
  }

  # Convert to DataFrame
  df_input = pd.DataFrame([input_data])

  # One-hot encoding matching your training pipeline
  df_input = pd.get_dummies(
      df_input,
      columns=[
          "Item_Fat_Content",
          "Item_Type",
          "Outlet_Size",
          "Outlet_Location_Type",
          "Outlet_Type",
      ],
      dtype=int,
  )

  # Reindex to ensure exact match with training columns (fill missing dummy columns with 0)
  df_input = df_input.reindex(columns=model_columns, fill_value=0)

  # Scale the data using saved scaler
  scaled_input = scaler.transform(df_input)

  # Make Prediction
  prediction = model.predict(scaled_input)[0]

  # Display Result
  st.success(f"Predicted Item Outlet Sales: **$ {prediction:,.2f}** 💰")