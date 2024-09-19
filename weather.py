import requests

# Replace 'YOUR_API_KEY' with your actual API key
api_key = 'a4b77f489ab8213a692d6335356bd924'
city = 'bengaluru'  # Replace with the name of the city you want to get weather data for

# API endpoint URL for current weather data
url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'

# Make a GET request to the API
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    
    # Extract relevant weather data
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    wind_speed = data['wind']['speed']
    
    # Print the weather data
    print(f"Temperature: {temperature}°C")
    print(f"Humidity: {humidity}%")
    print(f"Wind Speed: {wind_speed} m/s")
else:
    # Print an error message if the request was not successful
    print(f"Error: Unable to retrieve weather data. Status code: {response.status_code}")
