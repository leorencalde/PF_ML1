from fastapi import FastAPI, Query
from pydantic import BaseModel  # Asegúrate de importar BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Cargar el modelo de machine learning
model = joblib.load('taxi_demand_model.joblib')

@app.get("/get_weather")
def get_weather(date: str = Query(..., description="Fecha para obtener datos climáticos en formato YYYY-MM-DD")):
    # Cargar los datos del clima desde un archivo Parquet
    df_clima = pd.read_parquet('')

    # Filtrar los datos para la fecha específica
    clima_seleccionado = df_clima[df_clima['date'] == date]

    # Verificar si existen datos para la fecha especificada
    if clima_seleccionado.empty:
        return {"error": "No hay datos disponibles para la fecha proporcionada."}

    # Seleccionar las columnas relevantes
    clima_resultado = clima_seleccionado[['date', 'weather_code', 'temperature_2m_max', 'temperature_2m_min', 
                                          'rain_sum', 'snowfall_sum', 'precipitation_hours', 'wind_speed_10m_max']]

    # Convertir el DataFrame a diccionario para retornar como JSON
    return clima_resultado.to_dict(orient="records")

# Esquema de los datos climáticos para el POST
class WeatherData(BaseModel):
    weather_code: float
    temperature_2m_max: float
    temperature_2m_min: float
    rain_sum: float
    snowfall_sum: float
    precipitation_hours: float
    wind_speed_10m_max: float

@app.post("/predict")
def predict_demand(data: WeatherData):
    # Crear un DataFrame a partir de los datos de entrada
    input_data = pd.DataFrame([data.dict()])

    # Realizar la predicción
    prediction = model.predict(input_data)
    
    return {
        "predicted_demand": prediction[0]
    }

# Ejecutar la aplicación con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
