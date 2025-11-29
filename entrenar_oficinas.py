# entrenar_oficinas_escalable.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from datetime import datetime

np.random.seed(42)
data = []
año_actual = 2025  # Ajusta según el año actual

for _ in range(1800):
    # Generar datos realistas
    numero_pisos = np.random.choice(
        ["1", "2", "3", "4", "5-10", "11-20", ">20"],
        p=[0.2, 0.2, 0.2, 0.15, 0.1, 0.08, 0.07]
    )
    
    area_por_piso = np.random.choice(
        ["<200", "200-400", "400-560", "560-1000", "1000-2500", ">2500"],
        p=[0.25, 0.25, 0.2, 0.15, 0.1, 0.05]
    )
    
    area_total = np.random.choice(
        ["<500", "500-2000", "2000-5000", "5000-15000", ">15000"],
        p=[0.2, 0.3, 0.25, 0.15, 0.1]
    )
    
    # Año de conformidad (2015-2025)
    año_conformidad = np.random.randint(2015, 2026)
    antigüedad = año_actual - año_conformidad
    
    if antigüedad <= 1:
        antigüedad_str = "0-1"
    elif antigüedad <= 3:
        antigüedad_str = "2-3"
    elif antigüedad <= 5:
        antigüedad_str = "4-5"
    elif antigüedad <= 10:
        antigüedad_str = "6-10"
    else:
        antigüedad_str = ">10"
    
    tiene_conformidad = np.random.choice([True, False], p=[0.8, 0.2])
    tipo_conformidad = np.random.choice(
        ["Obra Nueva", "Remodelación", "Ampliación", "Cambio de Giro", "Sin Conformidad"],
        p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )
    
    tipo_ocupacion = np.random.choice(
        ["Uso Exclusivo (todo el edificio)", "Uso Compartido (piso/área específica)"],
        p=[0.4, 0.6]
    )
    
    if "Exclusivo" in tipo_ocupacion:
        itse = "No Aplica (uso exclusivo)"
    else:
        itse = np.random.choice(["Sí", "No"], p=[0.6, 0.4])
    
    piso_ubicacion = np.random.choice(
        ["PB", "1", "2", "3", "4", "5-10", ">10", "Todo el edificio"],
        p=[0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1]
    )
    
    uso_original = np.random.choice(
        ["Oficinas desde origen", "Adaptado a oficinas"],
        p=[0.6, 0.4]
    )
    
    ha_remodelado = np.random.choice([True, False], p=[0.3, 0.7])

    # === Aplicar reglas NORMATIVAS ===
    # Convertir área por piso a valor numérico para comparar con 560
    if area_por_piso == "<200":
        area_por_piso_num = 100
    elif area_por_piso == "200-400":
        area_por_piso_num = 300
    elif area_por_piso == "400-560":
        area_por_piso_num = 480
    elif area_por_piso == "560-1000":
        area_por_piso_num = 780
    elif area_por_piso == "1000-2500":
        area_por_piso_num = 1750
    else:
        area_por_piso_num = 3000
    
    # Convertir número de pisos a valor numérico
    if numero_pisos == "1":
        pisos_num = 1
    elif numero_pisos == "2":
        pisos_num = 2
    elif numero_pisos == "3":
        pisos_num = 3
    elif numero_pisos == "4":
        pisos_num = 4
    else:
        pisos_num = 6  # valor representativo para >4
    
    # Reglas de subfunción
    if area_por_piso_num > 560:
        subfuncion = "6.5"
    elif pisos_num <= 4 and area_por_piso_num <= 560:
        subfuncion = "6.1"
    elif tiene_conformidad and antigüedad <= 5 and (
        uso_original == "Oficinas desde origen" or 
        (ha_remodelado and tipo_conformidad in ["Remodelación", "Ampliación", "Cambio de Giro"])
    ):
        subfuncion = "6.2"
    elif "Compartido" in tipo_ocupacion:
        if itse == "Sí":
            subfuncion = "6.3"
        else:
            subfuncion = "6.4"
    else:
        # Casos ambiguos
        if area_por_piso_num > 560:
            subfuncion = "6.5"
        elif pisos_num <= 4:
            subfuncion = "6.1"
        else:
            subfuncion = "6.5"

    # === AÑADIR RUIDO (5%) ===
    # if np.random.rand() < 0.01:
    #     subfuncion = np.random.choice(["6.1", "6.2", "6.3", "6.4", "6.5"])

    # === CARACTERÍSTICAS ESCALABLES ===
    # 1. Cumple 6.1: ≤4 pisos y ≤560 m² por piso
    cumple_6_1 = 1 if (pisos_num <= 4 and area_por_piso_num <= 560) else 0
    
    # 2. Es 6.5: >560 m² por piso
    es_6_5 = 1 if area_por_piso_num > 560 else 0
    
    # 3. Conformidad reciente (≤5 años)
    conformidad_reciente = 1 if (tiene_conformidad and antigüedad <= 5) else 0
    
    # 4. Uso compartido
    uso_compartido = 1 if "Compartido" in tipo_ocupacion else 0
    
    # 5. ITSE vigente
    itse_vigente = 1 if itse == "Sí" else 0
    
    # 6. Área por piso numérica
    area_por_piso_num_final = area_por_piso_num
    
    # 7. Número de pisos numérico
    pisos_num_final = pisos_num
    
    # 8. Área total numérica
    area_total_map = {"<500": 300, "500-2000": 1250, "2000-5000": 3500, "5000-15000": 10000, ">15000": 20000}
    area_total_num = area_total_map.get(area_total, 1250)
    
    # 9. Año conformidad
    año_conformidad_final = año_conformidad
    
    # 10. Ha remodelado
    ha_remodelado_final = int(ha_remodelado)

    data.append([
        cumple_6_1,
        es_6_5,
        conformidad_reciente,
        uso_compartido,
        itse_vigente,
        area_por_piso_num_final,
        pisos_num_final,
        area_total_num,
        año_conformidad_final,
        ha_remodelado_final,
        subfuncion
    ])

# === Guardar y entrenar ===
df = pd.DataFrame(data, columns=[
    "cumple_6_1", "es_6_5", "conformidad_reciente", "uso_compartido", "itse_vigente",
    "area_por_piso", "numero_pisos", "area_total", "año_conformidad", "ha_remodelado",
    "subfuncion"
])

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/dataset_oficinas.csv", index=False)

X = df.drop("subfuncion", axis=1)
y = df["subfuncion"]

modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=12,
    random_state=42,
    class_weight="balanced"
)
modelo.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(modelo, "models/rf_oficinas.pkl")
print("✅ Modelo OFICINAS ADMINISTRATIVAS ESCALABLE guardado en models/rf_oficinas.pkl")