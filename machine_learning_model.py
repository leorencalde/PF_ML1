# machine_learning_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Cargar los datasets
# df_taxis = pd.read_parquet('ruta/al/dataset_taxis.parquet')
# df_clima = pd.read_csv('ruta/al/dataset_clima.csv')

# Simulación de la carga de datos
df_taxis = pd.DataFrame({'fecha_hora': pd.date_range(start='1/6/2023', periods=100, freq='h'),
                         'cantidad_viajes': [50]*100})
df_clima = pd.DataFrame({'fecha_hora': pd.date_range(start='1/6/2023', periods=100, freq='h'),
                         'temp': [20]*100, 'humedad': [50]*100, 'viento': [5]*100, 'precip': [0]*100})

# Combinar datasets basados en la fecha y hora
df = pd.merge(df_taxis, df_clima, on='fecha_hora')

# Variables y features
features = ['temp', 'humedad', 'viento', 'precip']
target = 'cantidad_viajes'

# División en entrenamiento y prueba
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['temp', 'humedad', 'viento', 'precip']),
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
