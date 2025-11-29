# entrenar_salud.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

np.random.seed(42)
data = []

# Definir mapeo de tipos (incluyendo "Otros" = 7)
tipo_map_inverso = {
    1: ["Puesto", "Posta"],
    2: ["Consultorio", "Consultorio médico"],
    3: ["Centro de salud", "Centro médico", "Policlínico", "Centro médico especializado"],
    4: ["Hospital general"],
    5: ["Hospital especializado"],
    6: ["Instituto"],
    7: ["Clínica móvil", "Hospital naval", "Unidad comunitaria", "Centro geriátrico", "Dispensario"]  # ← Nuevos tipos
}

# Generar datos sintéticos con "Otros"
for _ in range(2000):
    # Elegir clase base
    clase_base = np.random.choice(["I-1", "I-2", "I-3", "I-4", "II", "III"], p=[0.18, 0.18, 0.18, 0.14, 0.18, 0.14])
    
    # Decidir si es "Otros" (7) o no
    if np.random.rand() < 0.08:  # 8% de los casos son "Otros"
        tipo = 7
        # Para "Otros", asignar características realistas según la subfunción
        if clase_base in ["I-1", "I-2"]:
            nivel = 1
            camas = 0
            no_autosuf = False if clase_base == "I-1" else np.random.choice([True, False], p=[0.3, 0.7])
            capacidad = np.random.choice([1, 2])
            num_servicios = np.random.randint(1, 3)
            urg24 = False
            esp = 0
            pisos = 1
            area = np.random.uniform(50, 200)
            personal = np.random.randint(1, 5)
        elif clase_base == "I-3":
            nivel = 1
            camas = 0
            no_autosuf = True
            capacidad = 3
            num_servicios = np.random.randint(2, 5)
            urg24 = True
            esp = 1
            pisos = np.random.choice([1, 2])
            area = np.random.uniform(300, 1500)
            personal = np.random.randint(5, 20)
        elif clase_base == "I-4":
            nivel = 1
            camas = np.random.choice([1, 2])  # 1-10 o 11-50
            no_autosuf = True
            capacidad = 3
            num_servicios = np.random.randint(3, 5)
            urg24 = True
            esp = 1
            pisos = 2
            area = np.random.uniform(800, 2500)
            personal = np.random.randint(10, 30)
        elif clase_base == "II":
            nivel = 2
            camas = np.random.choice([2, 3])
            no_autosuf = True
            capacidad = 3
            num_servicios = 4
            urg24 = True
            esp = 1
            pisos = np.random.choice([2, 3])
            area = np.random.uniform(2000, 8000)
            personal = np.random.randint(20, 100)
        else:  # III
            nivel = 3
            camas = 3
            no_autosuf = True
            capacidad = 3
            num_servicios = 5
            urg24 = True
            esp = 2
            pisos = 3
            area = np.random.uniform(5000, 20000)
            personal = np.random.randint(50, 300)
    else:
        # Casos normales (no "Otros")
        if clase_base == "I-1":
            nivel, tipo, camas, no_autosuf, capacidad, num_servicios, urg24, esp, pisos, area, personal = (
                1, 1, 0, False, np.random.choice([1,2]), np.random.randint(0,2), False, 0, 1, np.random.uniform(40,100), np.random.randint(1,3)
            )
        elif clase_base == "I-2":
            nivel, tipo, camas, no_autosuf, capacidad, num_servicios, urg24, esp, pisos, area, personal = (
                1, 2, 0, False, np.random.choice([2,3]), np.random.randint(1,3), np.random.choice([True,False],p=[0.3,0.7]), 0, 1, np.random.uniform(60,120), 1
            )
        elif clase_base == "I-3":
            nivel, tipo, camas, no_autosuf, capacidad, num_servicios, urg24, esp, pisos, area, personal = (
                1, 3, 0, True, 3, np.random.randint(2,5), True, 1, np.random.choice([1,2]), np.random.uniform(500,1500), np.random.randint(8,20)
            )
        elif clase_base == "I-4":
            nivel, tipo, camas, no_autosuf, capacidad, num_servicios, urg24, esp, pisos, area, personal = (
                1, 3, np.random.choice([1,2]), True, 3, np.random.randint(3,5), True, 1, 2, np.random.uniform(1000,2500), np.random.randint(15,30)
            )
        elif clase_base == "II":
            nivel, tipo, camas, no_autosuf, capacidad, num_servicios, urg24, esp, pisos, area, personal = (
                2, 4, np.random.choice([2,3]), True, 3, 4, True, 1, np.random.choice([2,3]), np.random.uniform(3000,8000), np.random.randint(40,100)
            )
        else:  # III
            nivel, tipo, camas, no_autosuf, capacidad, num_servicios, urg24, esp, pisos, area, personal = (
                3, np.random.choice([5,6]), 3, True, 3, 5, True, 2, 3, np.random.uniform(8000,20000), np.random.randint(100,300)
            )
    
    data.append([
        nivel, tipo, camas, int(no_autosuf), capacidad,
        num_servicios, int(urg24), esp, pisos,
        area, personal, clase_base
    ])

# Crear DataFrame
df = pd.DataFrame(data, columns=[
    "nivel_atencion", "tipo_establecimiento", "camas_internamiento",
    "usuarios_no_autosuficientes", "capacidad_atencion", "num_servicios",
    "urgencias_24h", "num_especialidades", "num_pisos",
    "area_construida", "personal_medico_total", "subfuncion"
])

# Guardar CSV
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/dataset_salud.csv", index=False)

# Entrenar modelo con regularización
X = df.drop("subfuncion", axis=1)
y = df["subfuncion"]

modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)
modelo.fit(X, y)

# Guardar modelo
os.makedirs("models", exist_ok=True)
joblib.dump(modelo, "models/rf_salud.pkl")
print("✅ Modelo escalable guardado. Ahora acepta nuevos tipos de establecimiento.")