# entrenar_hospedaje.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

np.random.seed(42)
data = []

# Tipos especiales mencionados en 3.1
tipos_3_1 = {"ecolodge", "albergue"}

for _ in range(1600):
    # Generar datos realistas
    categoria_estrellas = np.random.choice([0,1,2,3,4,5], p=[0.1, 0.2, 0.25, 0.25, 0.1, 0.1])
    tipo_hospedaje = np.random.choice(["hotel", "hostal", "albergue", "ecolodge", "apart_hotel"])
    num_pisos = np.random.randint(1, 10)
    tiene_sotano = np.random.choice([True, False], p=[0.3, 0.7])
    num_habitaciones = np.random.randint(5, 300)
    capacidad_ocupantes = np.random.randint(10, 800)
    uso_mixto = np.random.choice([True, False], p=[0.4, 0.6])
    tiene_estacionamiento = np.random.choice([True, False], p=[0.6, 0.4])
    estacionamiento_en_sotano = tiene_estacionamiento and np.random.choice([True, False], p=[0.7, 0.3])

    # Calcular área de estacionamiento (simulada)
    area_estacionamiento = 0
    if estacionamiento_en_sotano:
        area_estacionamiento = np.random.uniform(200, 1200)

    # Aplicar reglas NORMATIVAS (prioridad fija)
    if estacionamiento_en_sotano and area_estacionamiento > 500:
        subfuncion = "3.4"
    elif num_pisos > 4:
        subfuncion = "3.3"
    elif categoria_estrellas <= 3 and num_pisos <= 4 and (tipo_hospedaje in tipos_3_1 or not tiene_sotano):
        subfuncion = "3.1"
    elif categoria_estrellas <= 3 and num_pisos <= 4 and tiene_sotano:
        subfuncion = "3.2"
    else:
        # Casos ambiguos → asignar según características dominantes
        if num_pisos > 4:
            subfuncion = "3.3"
        elif estacionamiento_en_sotano and area_estacionamiento > 500:
            subfuncion = "3.4"
        elif tiene_sotano:
            subfuncion = "3.2"
        else:
            subfuncion = "3.1"

    # Añadir ruido (5% de casos con subfunción cambiada)
    if np.random.rand() < 0.05:
        subfuncion = np.random.choice(["3.1", "3.2", "3.3", "3.4"])

    data.append([
        categoria_estrellas,
        1 if tipo_hospedaje in ["ecolodge", "albergue"] else 0,  # tipo_especial
        num_pisos,
        int(tiene_sotano),
        num_habitaciones,
        capacidad_ocupantes,
        int(uso_mixto),
        int(tiene_estacionamiento),
        int(estacionamiento_en_sotano),
        area_estacionamiento,  # característica adicional para 3.4
        subfuncion
    ])

df = pd.DataFrame(data, columns=[
    "categoria_estrellas", "tipo_especial", "num_pisos", "tiene_sotano",
    "num_habitaciones", "capacidad_ocupantes", "uso_mixto",
    "tiene_estacionamiento", "estacionamiento_en_sotano", "area_estacionamiento",
    "subfuncion"
])

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/dataset_hospedaje.csv", index=False)

# Entrenar modelo
X = df.drop("subfuncion", axis=1)
y = df["subfuncion"]

modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    min_samples_split=10,
    random_state=42,
    class_weight="balanced"
)
modelo.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(modelo, "models/rf_hospedaje.pkl")
print("✅ Modelo HOSPEDAJE guardado en models/rf_hospedaje.pkl")