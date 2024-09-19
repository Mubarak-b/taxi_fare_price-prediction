import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, median_absolute_error
import numpy as np
import requests
import time

# Load the dataset
data = pd.read_csv('dataset.csv')

# Feature selection - considering only relevant features for prediction
features = ['hour', 'day', 'month', 'temperature', 'humidity', 'windSpeed', 'distance']
target = 'price'

# Drop source and destination columns
data.drop(['source', 'destination'], axis=1, inplace=True)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Model initialization and training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R²):", r2)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("Median Absolute Error:", medae)

# The rest of the code remains the same...

def get_coordinates(address, retries=3, delay=1):
    geolocator = Nominatim(user_agent="my_geocoder")
    for attempt in range(retries):
        try:
            location = geolocator.geocode(address, timeout=10)
            if location:
                return location.latitude, location.longitude
            else:
                print(f"Coordinates not found for the address: {address}")
                return None, None
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    print(f"Failed to get coordinates for {address} after {retries} attempts.")
    return None, None

def calculate_distance(source_address, destination_address):
    source_latitude, source_longitude = get_coordinates(source_address)
    destination_latitude, destination_longitude = get_coordinates(destination_address)
    if source_latitude is not None and source_longitude is not None and destination_latitude is not None and destination_longitude is not None:
        source_coords = (source_latitude, source_longitude)
        destination_coords = (destination_latitude, destination_longitude)
        distance = geodesic(source_coords, destination_coords).kilometers
        print("You need to travel", distance * 1.45, "km to reach your destination")
        return distance * 1.45
    else:
        return None

def get_weather(latitude, longitude):
    api_key = "your_api_key"
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']
        print("Temperature:", temperature)
        print("Humidity:", humidity)
        print("Wind Speed:", wind_speed)
        return temperature, humidity, wind_speed
    else:
        print("Failed to retrieve weather data.")
        return 27, 50, 4.5

def get_user_input():
    source_address = input("Enter the source address: ")
    destination_address = input("Enter the destination address: ")
    return source_address, destination_address

def predict_fare(source_address, destination_address, hour, day, month):
    distance = calculate_distance(source_address, destination_address)
    if distance is not None:
        source_latitude, source_longitude = get_coordinates(source_address)
        if source_latitude is not None and source_longitude is not None:
            temperature, humidity, wind_speed = get_weather(source_latitude, source_longitude)
            if temperature is not None and humidity is not None and wind_speed is not None:
                # Create a DataFrame with the instance data
                instance_data = {
                    'hour': [hour],
                    'day': [day],
                    'month': [month],
                    'temperature': [temperature],
                    'humidity': [humidity],
                    'windSpeed': [wind_speed],
                    'distance': [distance],
                }
                
                # Add one-hot encoded categorical features to the DataFrame
                for col in data.columns:
                    if col.startswith('source_') or col.startswith('destination_'):
                        instance_data[col] = [0]  # Initialize with zeros
                source_col = f'source_{source_address}'
                if source_col in data.columns:
                    instance_data[source_col] = [1]
                destination_col = f'destination_{destination_address}'
                if destination_col in data.columns:
                    instance_data[destination_col] = [1]
                
                instance_df = pd.DataFrame(instance_data)
                
                # Make sure day and month columns are included
                if 'day' not in instance_df.columns:
                    instance_df['day'] = 0
                if 'month' not in instance_df.columns:
                    instance_df['month'] = 0
                
                fare_prediction = model.predict(instance_df)
                return fare_prediction[0]
    return None

# Example usage:
def main():
    source_address, destination_address = get_user_input()
    hour = int(input("Enter the hour: "))
    day = int(input("Enter the day: "))
    month = int(input("Enter the month: "))

    predicted_fare = predict_fare(source_address, destination_address, hour, day, month)
    if predicted_fare is not None:
        print("Predicted Fare:", predicted_fare)
    else:
        print("Failed to predict fare.")

if __name__ == "__main__":
    main()
