import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import streamlit as st

# Create a simple dataset for demonstration
data = {
    'Size': [1500, 2000, 2500, 3000, 3500],
    'Bedrooms': [3, 4, 3, 5, 4],
    'Age': [10, 15, 20, 5, 8],
    'Price': [300000, 400000, 500000, 600000, 700000]
}
house_data = pd.DataFrame(data)

# Define features and target variable
X = house_data[['Size', 'Bedrooms', 'Age']]
y = house_data['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a Pickle file
pickle_file_path = 'house_price_model.pkl'
with open(pickle_file_path, 'wb') as model_file:
    pickle.dump(model, model_file)

# Streamlit application
st.title("House Price Prediction Model")

# Provide a download button for the Pickle file
with open(pickle_file_path, 'rb') as f:
    st.download_button('Download Trained Model', f, file_name='house_price_model.pkl', mime='application/octet-stream')

# Optional: Add functionality to predict house prices based on user input
st.header("Predict House Price")
size = st.number_input("Enter the Size of the House (in sqft):", min_value=0)
bedrooms = st.number_input("Enter the Number of Bedrooms:", min_value=0)
age = st.number_input("Enter the Age of the House (in years):", min_value=0)

if st.button("Predict Price"):
    input_data = [[size, bedrooms, age]]
    predicted_price = model.predict(input_data)
    st.success(f"The predicted price of the house is: ${predicted_price[0]:,.2f}")
