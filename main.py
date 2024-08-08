import requests
from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

# Cargar el modelo de machine learning
model = joblib.load('./taxi_demand_model.joblib')

app = FastAPI()

# Definir el esquema de la solicitud
class WeatherData(BaseModel):
    city: str

@app.get("/predict")
def prediccion_demanda_taxis(city: str = Query(..., description="Nombre de la ciudad para obtener datos climáticos")):
    # Obtener los datos del clima de la API de OpenWeather
    api_key = "b0246ce45acfb5b01b3600e57b12b1a9"  # Reemplaza esto con tu clave de API real
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    weather_data = response.json()
    
    temp = weather_data['main']['temp']
    humedad = weather_data['main']['humidity']
    viento = weather_data['wind']['speed']
    precip = weather_data.get('rain', {}).get('1h', 0)
    
    # Crear un DataFrame a partir de los datos de entrada
    input_data = pd.DataFrame([{
        'temp': temp,
        'humedad': humedad,
        'viento': viento,
        'precip': precip
    }])

    # Realizar la predicción
    prediction = model.predict(input_data)
    
    # Obtener la fecha actual
    fecha_consulta = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return {
        "weather_data": {
            "city": city,
            "date_of_query": fecha_consulta,
            "temp": temp,
            "humidity": humedad,
            "wind_speed": viento,
            "precipitation": precip
        },
        "predicted_demand": prediction[0]
    }

# Ejecutar la aplicación con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
