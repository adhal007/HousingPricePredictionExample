# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import streamlit as st

# Load and prepare the data
housing_data = pd.read_csv('/Users/abhilashdhal/OmixHub/Housing.csv')

# Select the most important features (area, bedrooms, bathrooms, stories, and mainroad)
selected_features = ['area', 'bedrooms', 'bathrooms', 'stories']
# Convert mainroad to numeric
housing_data['mainroad'] = housing_data['mainroad'].map({'yes': 1, 'no': 0})
selected_features.append('mainroad')

X = housing_data[selected_features]
y = housing_data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Create the Streamlit dashboard
st.title('House Price Prediction Dashboard')

# Create input fields for features
st.sidebar.header('Input House Features')
area = st.sidebar.number_input('Area (sq ft)', min_value=1000, max_value=15000, value=3000)
bedrooms = st.sidebar.number_input('Number of Bedrooms', min_value=1, max_value=6, value=2)
bathrooms = st.sidebar.number_input('Number of Bathrooms', min_value=1, max_value=4, value=2)
stories = st.sidebar.number_input('Number of Stories', min_value=1, max_value=4, value=1)
mainroad = st.sidebar.selectbox('Main Road', ['yes', 'no'])

# Convert inputs to model format
input_data = np.array([[
    area,
    bedrooms,
    bathrooms,
    stories,
    1 if mainroad == 'yes' else 0
]])

# Scale the input
input_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_scaled)[0]

# Display prediction
st.header('Predicted House Price')
st.write(f'₹{prediction:,.2f}')

# After the prediction section, add distribution visualization
st.header('Price Distribution by Bedrooms')

# Create distribution plot
import plotly.figure_factory as ff
import plotly.graph_objects as go

# Get prices for houses with same number of bedrooms
similar_houses = housing_data[housing_data['bedrooms'] == bedrooms]['price']

# Create distribution plot
fig = go.Figure()

# Add distribution curve
fig.add_trace(go.Histogram(
    x=similar_houses,
    name='Price Distribution',
    nbinsx=30,
    histnorm='probability density'
))

# Add vertical line for prediction
fig.add_vline(
    x=prediction,
    line_dash="dash",
    line_color="red",
    annotation_text="Predicted Price",
    annotation_position="top"
)

# Customize layout
fig.update_layout(
    title=f'Price Distribution for {bedrooms} Bedroom Houses',
    xaxis_title='Price (₹)',
    yaxis_title='Density',
    showlegend=False
)

# Display the plot
st.plotly_chart(fig)

# Add some statistics
st.write('Statistics for houses with the same number of bedrooms:')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Average Price', f'₹{similar_houses.mean():,.2f}')
with col2:
    st.metric('Median Price', f'₹{similar_houses.median():,.2f}')
with col3:
    st.metric('Your Prediction', f'₹{prediction:,.2f}')
    
    
# Display model performance metrics
st.sidebar.header('Model Performance')
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
st.sidebar.write(f'Training R² Score: {train_score:.3f}')
st.sidebar.write(f'Testing R² Score: {test_score:.3f}')
