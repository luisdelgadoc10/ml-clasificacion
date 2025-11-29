# entrenar_almacen_escalable.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

np.random.seed(42)
data = []

# Palabras clave para 8.3 (explosivos/pirotécnicos)
palabras_8_3 = {"explosivo", "pirotécnico", "municion", "fuegos artificiales", "pólvora"}

for _ in range(1500):
    # Generar datos realistas con valores nuevos
    tipo_cobertura = np.random.choice(
        ["No Techado", "Parcialmente Techado", "Totalmente Techado", "Cerrado y Techado"],
        p=[0.25, 0.2, 0.3, 0.25]
    )
    
    porcentaje_techado = np.random.choice(
        ["0%", "1-25%", "26-50%", "51-75%", "76-99%", "100%"],
        p=[0.2, 0.15, 0.15, 0.15, 0.15, 0.2]
    )
    
    tipo_cerramiento = np.random.choice(
        ["Abierto", "Semi-abierto (muros parciales)", "Cerrado (muros completos)", "Con climatización"],
        p=[0.25, 0.25, 0.3, 0.2]
    )
    
    tipo_establecimiento = np.random.choice(
        ["Almacén General", "Depósito", "Centro de Distribución", "Estacionamiento Vehicular", "Almacén Especializado", "Bodega", "Terminal logístico"],
        p=[0.25, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05]
    )
    
    uso_principal = np.random.choice(
        ["Almacenamiento de mercancías", "Estacionamiento de vehículos", "Mixto (almacén + estacionamiento)", "Depósito temporal", "Centro logístico"],
        p=[0.4, 0.25, 0.2, 0.1, 0.05]
    )
    
    tipo_productos = np.random.choice(
        ["Ninguno (vacío/vehículos)", "Productos no peligrosos", "Inflamables Clase IIIA (punto inflamación >60°C)", 
         "Inflamables Clase I-II (punto inflamación <60°C)", "Explosivos", "Pirotécnicos", "Municiones", "Materiales relacionados explosivos", "Productos generales"],
        p=[0.3, 0.3, 0.15, 0.1, 0.04, 0.04, 0.03, 0.02, 0.02]
    )
    
    almacena_explosivos = (
        tipo_productos in ["Explosivos", "Pirotécnicos", "Municiones", "Materiales relacionados explosivos"] or
        np.random.rand() > 0.95
    )
    
    nivel_nfpa = np.random.choice(
        ["0 (mínimo)", "1 (ligero)", "2 (moderado)", "3 (serio)", "4 (severo)"],
        p=[0.4, 0.25, 0.2, 0.1, 0.05]
    )
    
    tiene_areas_admin = np.random.choice([True, False], p=[0.6, 0.4])
    
    area_admin = np.random.choice(
        ["0", "1-50", "51-200", "201-500", ">500"],
        p=[0.3, 0.25, 0.25, 0.15, 0.05]
    )

    # === Aplicar reglas NORMATIVAS (en orden de prioridad) ===
    if almacena_explosivos or any(p in tipo_productos.lower() for p in palabras_8_3):
        subfuncion = "8.3"
    elif tipo_cobertura == "No Techado" or porcentaje_techado == "0%":
        subfuncion = "8.1"
    else:
        subfuncion = "8.2"

    # === AÑADIR RUIDO (5%) ===
    # if np.random.rand() < 0.01:
    #     subfuncion = np.random.choice(["8.1", "8.2", "8.3"])

    # === CARACTERÍSTICAS ESCALABLES ===
    # 1. Es 8.3: productos peligrosos
    es_8_3 = 1 if (almacena_explosivos or any(p in tipo_productos.lower() for p in palabras_8_3)) else 0
    
    # 2. Es 8.1: no techado
    es_8_1 = 1 if (tipo_cobertura == "No Techado" or porcentaje_techado == "0%") else 0
    
    # 3. Porcentaje techado numérico
    porcentaje_map = {"0%": 0, "1-25%": 15, "26-50%": 37, "51-75%": 62, "76-99%": 87, "100%": 100}
    porcentaje_num = porcentaje_map.get(porcentaje_techado, 50)
    
    # 4. Tipo cobertura numérico
    cobertura_map = {
        "No Techado": 0,
        "Parcialmente Techado": 1,
        "Totalmente Techado": 2,
        "Cerrado y Techado": 3
    }
    cobertura_num = cobertura_map.get(tipo_cobertura, 1)
    
    # 5. Tipo cerramiento numérico
    cerramiento_map = {
        "Abierto": 0,
        "Semi-abierto (muros parciales)": 1,
        "Cerrado (muros completos)": 2,
        "Con climatización": 3
    }
    cerramiento_num = cerramiento_map.get(tipo_cerramiento, 1)
    
    # 6. Nivel NFPA numérico
    nfpa_map = {
        "0 (mínimo)": 0,
        "1 (ligero)": 1,
        "2 (moderado)": 2,
        "3 (serio)": 3,
        "4 (severo)": 4
    }
    nfpa_num = nfpa_map.get(nivel_nfpa, 0)
    
    # 7. Tiene áreas administrativas
    tiene_areas_admin_num = int(tiene_areas_admin)
    
    # 8. Área administrativa numérica
    area_admin_map = {"0": 0, "1-50": 30, "51-200": 125, "201-500": 350, ">500": 750}
    area_admin_num = area_admin_map.get(area_admin, 30)
    
    # 9. Es estacionamiento
    es_estacionamiento = 1 if ("estacionamiento" in uso_principal.lower() or "vehicular" in tipo_establecimiento.lower()) else 0

    data.append([
        es_8_3,
        es_8_1,
        porcentaje_num,
        cobertura_num,
        cerramiento_num,
        nfpa_num,
        tiene_areas_admin_num,
        area_admin_num,
        es_estacionamiento,
        subfuncion
    ])

# === Guardar y entrenar ===
df = pd.DataFrame(data, columns=[
    "es_8_3", "es_8_1", "porcentaje_techado", "tipo_cobertura", "tipo_cerramiento",
    "nivel_nfpa", "tiene_areas_admin", "area_admin", "es_estacionamiento",
    "subfuncion"
])

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/dataset_almacen.csv", index=False)

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
joblib.dump(modelo, "models/rf_almacen.pkl")
print("✅ Modelo ALMACÉN ESCALABLE guardado en models/rf_almacen.pkl")