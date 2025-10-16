# entrenar_educacion.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

np.random.seed(42)
data = []

for _ in range(1800):
    # === Generar datos realistas con valores nuevos ===
    nivel_educativo = np.random.choice(
        ["Inicial", "Primaria", "Secundaria", "Superior Técnico", "Superior Universitario", 
         "Bachillerato Internacional", "Centro de idiomas", "Academia técnica", "Escuela vocacional"],
        p=[0.18, 0.18, 0.18, 0.12, 0.12, 0.07, 0.05, 0.05, 0.05]
    )
    
    tipo_institucion = np.random.choice(
        ["CEBE", "Colegio Regular", "Instituto", "Escuela Superior", "Centro Superior", "Universidad",
         "Academia privada", "Centro cultural", "Institución técnica", "Colegio internacional"],
        p=[0.12, 0.20, 0.12, 0.10, 0.10, 0.12, 0.06, 0.04, 0.08, 0.06]
    )
    
    # Número de pisos real (1-12)
    num_pisos_real = np.random.randint(1, 13)
    
    # Área construida real (100-25000 m²)
    area_real = np.random.uniform(100, 25000)
    
    # CEBE y algunos colegios atienden discapacidad
    atiende_personas_discapacidad = (
        (tipo_institucion == "CEBE" and np.random.rand() > 0.3) or
        (tipo_institucion == "Colegio Regular" and np.random.rand() > 0.7) or
        (np.random.rand() > 0.85)
    )
    
    # Capacidad de alumnos real
    if "Superior" in nivel_educativo or tipo_institucion in ["Universidad", "Instituto", "Escuela Superior", "Centro Superior"]:
        cap_real = np.random.randint(200, 5000)
    else:
        cap_real = np.random.randint(30, 1500)
    
    cantidad_aulas = np.random.randint(5, 120)
    
    tipo_edificacion = np.random.choice(
        ["Construida como Educativa", "Remodelada/Acondicionada para Educación"],
        p=[0.75, 0.25]
    )

    # === Aplicar reglas NORMATIVAS para la etiqueta ===
    if "remodelada" in tipo_edificacion.lower() or "acondicionada" in tipo_edificacion.lower():
        subfuncion = "4.4"
    elif num_pisos_real > 3:
        subfuncion = "4.2"
    elif nivel_educativo in ["Inicial", "Primaria", "Secundaria"] and atiende_personas_discapacidad and num_pisos_real <= 3:
        subfuncion = "4.1"
    elif ("Superior" in nivel_educativo or 
          tipo_institucion in ["Instituto", "Escuela Superior", "Centro Superior", "Universidad"]):
        subfuncion = "4.3"
    else:
        # Casos ambiguos: educación básica sin discapacidad
        if num_pisos_real <= 3:
            subfuncion = "4.1"
        else:
            subfuncion = "4.2"

    # === AÑADIR RUIDO (5% de casos con etiqueta cambiada) ===
    if np.random.rand() < 0.05:
        subfuncion = np.random.choice(["4.1", "4.2", "4.3", "4.4"])

    # === GENERAR CARACTERÍSTICAS ESCALABLES (IGUALES QUE EN PREPROCESS) ===
    
    # 1. es_basico: 1 si es Inicial/Primaria/Secundaria
    es_basico = 1 if nivel_educativo in ["Inicial", "Primaria", "Secundaria"] else 0
    
    # 2. es_superior: 1 si es educación superior
    es_superior = 1 if (
        "Superior" in nivel_educativo or 
        tipo_institucion in ["Instituto", "Escuela Superior", "Centro Superior", "Universidad"]
    ) else 0
    
    # 3. num_pisos: ya es numérico
    # 4. area_num: ya es numérico
    # 5. discapacidad: 0/1
    # 6. cap_num: ya es numérico
    # 7. aulas: ya es numérico
    # 8. es_remoldeada: 1 si es remodelada
    es_remoldeada = 1 if "remodelada" in tipo_edificacion.lower() or "acondicionada" in tipo_edificacion.lower() else 0

    # Añadir a datos
    data.append([
        es_basico,
        es_superior,
        num_pisos_real,
        area_real,
        int(atiende_personas_discapacidad),
        cap_real,
        cantidad_aulas,
        es_remoldeada,
        subfuncion
    ])

# === Crear DataFrame ===
df = pd.DataFrame(data, columns=[
    "es_basico",
    "es_superior", 
    "num_pisos",
    "area_construida_m2",
    "atiende_personas_discapacidad",
    "capacidad_alumnos",
    "cantidad_aulas",
    "es_remoldeada",
    "subfuncion"
])

# === Guardar CSV ===
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/dataset_educacion.csv", index=False)

# === Entrenar modelo ===
X = df.drop("subfuncion", axis=1)
y = df["subfuncion"]

modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=12,
    min_samples_leaf=6,
    random_state=42,
    class_weight="balanced"
)
modelo.fit(X, y)

# === Guardar modelo ===
os.makedirs("models", exist_ok=True)
joblib.dump(modelo, "models/rf_educacion.pkl")
print("✅ Modelo EDUCACIÓN ESCALABLE guardado en models/rf_educacion.pkl")
