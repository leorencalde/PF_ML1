# machine_learning_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Cargar los datasets
df_taxis = pd.read_parquet('dataset_taxis.parquet')
df_clima = pd.read_parquet('dataset_clima.parquet')

# Asegúrate de que ambas columnas 'date' estén en el formato datetime
df_taxis['date'] = pd.to_datetime(df_taxis['date'])
df_clima['date'] = pd.to_datetime(df_clima['date'])

# Combinar datasets basados en la fecha y hora
df = pd.merge(df_taxis, df_clima, on='date')

# Variables y features
features = ['weather_code', 'temperature_2m_max', 'temperature_2m_min', 'rain_sum', 'snowfall_sum', 'precipitation_hours', 'wind_speed_10m_max']
target = 'demand'

# División en entrenamiento y prueba
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['weather_code', 'temperature_2m_max', 'temperature_2m_min', 'rain_sum', 'snowfall_sum', 'precipitation_hours', 'wind_speed_10m_max']),
    ])

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Entrenamiento del modelo
pipeline.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(pipeline, 'taxi_demand_model.joblib')
