import pandas as pd
import numpy as np
import pickle
import streamlit as st
from xgboost import XGBRegressor
import os





file_path = os.path.join(os.path.dirname(__file__), 'housing_data.csv')
price_df = pd.read_csv(file_path)

#price_df = pd.read_csv(r"C:\Users\ggpal\Downloads\housing_data.csv")
#print(price_df.head())

BASE_DIR = os.path.dirname(__file__)

# Paths to your files
features_path = os.path.join(BASE_DIR, 'features.pkl')
model_path = os.path.join(BASE_DIR, 'model.pkl')
options_path = os.path.join(BASE_DIR, 'location_opitions.pkl')

# Load the files
with open(features_path, 'rb') as file:
    features = pickle.load(file)

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(options_path, 'rb') as file:
    options = pickle.load(file)
    options = sorted(list(set(options)))


# -- Average Income based off of user input for Median Household Income column
def get_avg_income(city, county, state, zip_code):
    """Get Average Income of an Area Based off of Input"""
    avg_inc = price_df[
        (price_df['City'] == city)
        &
        (price_df['County'] == county)
        &
        (price_df['State'] == state)
        &
        (price_df['Zip Code'] == int(zip_code))
    ]['Median Household Income'].median()

    return avg_inc

# -- Gt Zip pop and density based on location
def get_zip_pop_density(city, county, state, zip_code):
    """Get Zip Code Population and Zip Code Density Based on User Input"""
    df = price_df[
        (price_df['City'] == city)
        &
        (price_df['County'] == county)
        &
        (price_df['State'] == state)
        &
        (price_df['Zip Code'] == int(zip_code))
    ]

    return df['Zip Code Density'].unique()[0], df['Zip Code Population'].unique()[0]

# -- Get lat and lng
def get_lat_lng(city, county, state, zip_code):
    """Get latitude and longitude Based on User Input"""
    df = price_df[
        (price_df['City'] == city)
        &
        (price_df['County'] == county)
        &
        (price_df['State'] == state)
        &
        (price_df['Zip Code'] == int(zip_code))
    ]

    return df['Latitude'].unique()[0], df['Longitude'].unique()[0]


# -- Fill In State Features for inputs in model
def fill_state_cols(state):
    """Based off of State input, fill in state inputs"""
    state_cols = [col for col in features if col.endswith('-State')]
    ls = [0] * len(state_cols)

    for i, temp_state in enumerate(state_cols):
        if temp_state[:temp_state.rfind('-')] == state:
            ls[i] = 1
            return ls

    return ls

# -- Fill In City Features for inputs in model
def fill_city_cols(city):
    """Based off of City Input, Fill in City Inputs"""    
    city_cols = [col for col in features if col.endswith('-City')]
    ls = [0] * len(city_cols)

    for i, temp_city in enumerate(city_cols):
        if temp_city[:temp_city.rfind('-')] == city:
            ls[i] = 1
            return ls
        
    return ls

# -- Fill In County Features for inputs in model
def fill_county_cols(county):
    """Based off of City Input, Fill in City Inputs"""    
    county_cols = [col for col in features if col.endswith('-County')]
    ls = [0] * len(county_cols)

    for i, temp_county in enumerate(county_cols):
        if temp_county[:temp_county.rfind('-')] == county:
            ls[i] = 1
            return ls
        
    return ls

# -- Fill In Zip Code Features for inputs in model
def fill_zip_code_cols(zip_code):
    """Fill in Zip Code Features Based off of input"""
    zip_code = str(zip_code)
    zip_code_cols = [col for col in features if col.endswith('-Zip_Code')]
    ls = [0] * len(zip_code_cols)

    for i, temp_code in enumerate(zip_code_cols):
        if temp_code[:temp_code.rfind('-')] == zip_code:
            ls[i] = 1
            return ls
        
    return ls 

# Predict home price
def predict(features):
    features = np.array(features)
    features = features.reshape(1, -1)

    pred = model.predict(features)
    return pred

# displat avg home price of an area
def avg_price(state, city, county, zip_code):
    temp = price_df[
        (price_df['State'] == state)
        &
        (price_df['City'] == city)
        &
        (price_df['County'] == county)
        &
        (price_df['Zip Code'] == int(zip_code))
    ]

    return temp.Price.mean()

# -- How many baths and beds does a usual home have
def med_beds_baths(state, city, county, zip_code):
    temp = price_df[
        (price_df['State'] == state)
        &
        (price_df['City'] == city)
        &
        (price_df['County'] == county)
        &
        (price_df['Zip Code'] == int(zip_code))
    ]

    return temp.Beds.median(), temp.Baths.median()

# Typical size of a home
def avg_size(state, city, county, zip_code):
    temp = price_df[
        (price_df['State'] == state)
        &
        (price_df['City'] == city)
        &
        (price_df['County'] == county)
        &
        (price_df['Zip Code'] == int(zip_code))
    ]

    return temp['Living Space'].mean()

# Streamlit app 
def main():
    st.set_page_config(page_title="🏡 Real Estate Price Estimator with XGBoost", layout="centered")
    st.title("🏠 Real Estate Price Estimator")
    #st.image("https://media2.dev.to/dynamic/image/width=1000,height=420,fit=cover,gravity=auto,format=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F7w8rh2oj5arc1epo2sls.png")
    
    st.markdown(
        "<div style='text-align: center;'>"
        "<img src='https://media2.dev.to/dynamic/image/width=1000,height=420,fit=cover,gravity=auto,format=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F7w8rh2oj5arc1epo2sls.png' width='400'/>"
        "</div>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "Enter the home's size, number of bedrooms and bathrooms, and its location to get an estimated price prediction.\n\n"
        "**Note:** This model was trained on a sample dataset from Kaggle and may not reflect real-world market accuracy.\n"
        "The model achieves an R² score of **0.75**, indicating moderate predictive performance."
    )
    # --- Form for Inputs ---
    with st.form("user_inputs"):
        st.subheader("🚪 House Features")
        living_space = st.number_input("Living Space (sqft)", min_value=500, step=50, format="%d")
        beds = st.number_input("Bedrooms", min_value=1, step=1)
        baths = st.number_input("Bathrooms", min_value=1, step=1)

        st.subheader("📍 Location")
        location = st.selectbox("Choose an option(City - County - State - Zip Code):", options)

        city, county, state, zip_code = location.split(' - ')


        submitted = st.form_submit_button("Next ➡️")

    if submitted:
        zip_dens, zip_pop = get_zip_pop_density(city, county, state, zip_code)
        lat, lng = get_lat_lng(city, county, state, zip_code)

        state_feats = fill_state_cols(state)
        city_feats = fill_city_cols(city)
        county_feats = fill_county_cols(county)
        zip_code_feats = fill_zip_code_cols(zip_code)

        # Create Features List
        all_features = [living_space, beds, baths] + [get_avg_income(city, county, state, zip_code)] + [zip_dens, zip_pop] + [lat, lng] + state_feats + city_feats + county_feats + zip_code_feats
        #st.success(f"Input received:\n\n- {living_space} sqft\n- {beds} beds\n- {baths} baths\n- Location: {location}\n - City: {city}, County: {county}, State: {state}, Zip Code: {zip_code}\n\n- Features: {all_features}\n- {all_features.count(1)}")

        st.success(f"Predicted House Price for a Home in {city}, {state}, {county} County\n- SQFT: {living_space}\n- Beds: {beds}\n- Baths: {baths}\n- Predction: ${predict(all_features)[0]:,.2f}")
        st.success(f"The Average Home Price in {city}, {state}, {county} County, {zip_code} is: ${avg_price(state, city, county, zip_code):,.2f}\n- Average Home Size: {np.round(avg_size(state, city, county, zip_code))}\n- Median Number of Beds: {med_beds_baths(state, city, county, zip_code)[0]}\n- Median Number of Baths: {med_beds_baths(state, city, county, zip_code)[1]}")
        

        map_data = pd.DataFrame({
            "Price": [predict(all_features)],
            'LAT': [lat],
            'LON': [lng]
        })

    

        st.map(data=map_data, zoom=4.5)

        #st.success(f"Input Received:\n\n- The Predicted House Price in {city}, {state}, {county} County - {zip_code} with a living space of {living_space} sq ft and {beds} beds and {baths} baths is ${predict(all_features):,.2f}")

if __name__ == '__main__':
    main()