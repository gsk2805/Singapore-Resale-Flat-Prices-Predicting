
import streamlit as st
import pandas as pd
from joblib import load
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load pre-trained models and encoders
df_copy = pd.read_csv(r'C:/Users/HP/Desktop/New project phyton/New project/ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv')
categorical_encoded = load('C:/Users/HP/Desktop/New project phyton/New project/categorical_encoded.joblib')
numerical_scaled = load('C:/Users/HP/Desktop/New project phyton/New project/numerical_scaled.joblib')
label_encoders = load('C:/Users/HP/Desktop/New project phyton/New project/label_encoders.joblib')
scaler_features = load('C:/Users/HP/Desktop/New project phyton/New project/scaler_features.joblib')
scaler_y = load('C:/Users/HP/Desktop/New project phyton/New project/scaler_target.joblib')
cat_regressor = load('C:/Users/HP/Desktop/New project phyton/New project/cat_regressor_model.joblib')

# Streamlit app configuration
st.set_page_config(page_title="Resale Flat Price Predictor", layout="wide")

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home","Prediction"])

if page == "Home":
    # Home page content
    st.title("Singapore Resale Flat Prices Predicting")
    st.write("""
        ## About the Project
        This project aims to predict the resale price of flats in Singapore based on various features such as 
        town, flat type, street name, storey range, flat model, floor area, and transaction details.
        
        The dataset used for this project includes resale flat prices from different time periods and contains 
        detailed information about each flat. The data was preprocessed and various machine learning models were 
        trained to predict the resale price accurately.
        
        The best-performing model, CatBoost Regressor, is used in this application to provide users with reliable 
        price predictions based on their input features.
             
    
    
        
        ### How to Use the Predictor
        Navigate to the "Prediction" page using the sidebar. Enter the required details about the flat, and the 
        application will provide the predicted resale price.
    """) 
    
    st.write("""GITHUB: "https://github.com/gsk2805/Singapore-Resale-Flat-Prices-Predicting""")
    st.write("""LINKEDIN": "https://www.linkedin.com/feed/update/urn:li:activity:7258916890255753219/""")
        
      
    

elif page == "Prediction":
    # Prediction page content
    st.title("Resale Flat Price Predictor")
    st.write("Enter the details of the flat to get the predicted resale price.")

    # Input features
    town = st.selectbox('Town', df_copy['town'].unique(), key='town_selectbox')
    flat_type = st.selectbox('Flat Type', df_copy['flat_type'].unique(), key='flat_type_selectbox')
    street_name = st.selectbox('Street Name', df_copy['street_name'].unique(), key='street_name_selectbox')
    storey_range = st.selectbox('Storey Range', df_copy['storey_range'].unique(), key='storey_range_selectbox')
    flat_model = st.selectbox('Flat Model', df_copy['flat_model'].unique(), key='flat_model_selectbox')
    month_month = st.slider('Month', 1, 12, key='month_slider')
    floor_area_sqm = st.number_input('Floor Area (sqm)', min_value=0, key='floor_area_input')
    month_year = st.number_input('Year of Transaction', min_value=1990, max_value=2030, step=1, key='transaction_year_input')
    lease_commence_date_year = st.number_input('Lease Commence Year', min_value=1990, max_value=2024, step=1, key='lease_commence_year_input')

    # Button to trigger prediction
    if st.button('Predict Price'):
        # Create input dataframe
        input_data = pd.DataFrame({
            'town': [town],
            'flat_type': [flat_type],
            'street_name': [street_name],
            'storey_range': [storey_range],
            'flat_model': [flat_model],
            'month_month': [month_month],
            'floor_area_sqm': [floor_area_sqm],
            'month_year': [month_year],
            'lease_commence_date_year': [lease_commence_date_year]
        })

        # Encode categorical features
        for col in categorical_encoded:
            input_data[col] = label_encoders[col].transform(input_data[col])

        # Scale numerical features
        input_data[numerical_scaled] = scaler_features.transform(input_data[numerical_scaled])

        # Predict resale price
        prediction_scaled = cat_regressor.predict(input_data)
        prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))

        # Display the prediction
        st.subheader("Predicted Resale Price")
        st.write(f"${prediction[0][0]*1000:,.2f}")  # Scale back to original price range

# Run the Streamlit app
if __name__ == "__main__":
    if page == "Home":
        st.write("Navigate to 'Prediction' to get started.")
    else:
        st.write("Fill in the details and click 'Predict Price' to get the predicted resale price.")
