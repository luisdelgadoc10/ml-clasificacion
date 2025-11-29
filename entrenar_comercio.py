# entrenar_comercio_escalable.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

np.random.seed(42)
data = []

# Palabras clave para 7.6 (explosivos/pirotécnicos)
palabras_7_6 = {"explosivo", "pirotécnico", "municion", "fuegos artificiales", "pólvora"}

for _ in range(1700):
    # Generar datos realistas con valores nuevos
    numero_pisos = np.random.choice(
        ["1", "2", "3", "4", "5-10", ">10"],
        p=[0.25, 0.25, 0.2, 0.1, 0.1, 0.1]
    )
    
    area_total = np.random.choice(
        ["<300", "300-750", "750-2000", "2000-10000", ">10000"],
        p=[0.2, 0.25, 0.25, 0.2, 0.1]
    )
    
    area_venta = np.random.choice(
        ["<200", "200-500", "500-1500", "1500-5000", ">5000"],
        p=[0.25, 0.25, 0.25, 0.15, 0.1]
    )
    
    tipo_establecimiento = np.random.choice(
        ["Tienda Individual", "Módulo/Stand/Puesto", "Mercado Minorista", "Mercado Mayorista", 
         "Supermercado", "Tienda por Departamentos", "Galería Comercial", "Centro Comercial", 
         "Complejo Comercial", "Tienda especializada", "Local de servicios"],
        p=[0.2, 0.15, 0.1, 0.08, 0.12, 0.08, 0.07, 0.08, 0.07, 0.03, 0.02]
    )
    
    modalidad_operacion = np.random.choice(
        ["Independiente", "Módulo en edificio corporativo", "Áreas comunes edificio mixto"],
        p=[0.5, 0.3, 0.2]
    )
    
    uso_edificacion = np.random.choice(
        ["Comercial Exclusivo", "Mixto (comercio + vivienda/oficina)", "Solo Áreas Comunes"],
        p=[0.4, 0.4, 0.2]
    )
    
    tipo_licencia = np.random.choice(
        ["Individual", "Corporativa (galería/mercado)", "Sin licencia"],
        p=[0.6, 0.3, 0.1]
    )
    
    if "Módulo" in tipo_establecimiento or "Galería" in tipo_establecimiento or "Centro Comercial" in tipo_establecimiento:
        edificio_licencia = np.random.choice(["Sí", "No"], p=[0.7, 0.3])
    else:
        edificio_licencia = "No Aplica (establecimiento independiente)"
    
    tipo_productos = np.random.choice(
        ["Ninguno", "Explosivos", "Pirotécnicos", "Municiones", "Materiales relacionados", "Inflamables Clase I-II", "Productos generales"],
        p=[0.7, 0.05, 0.05, 0.04, 0.03, 0.08, 0.05]
    )
    
    comercializa_explosivos = (
        tipo_productos in ["Explosivos", "Pirotécnicos", "Municiones", "Materiales relacionados"] or
        np.random.rand() > 0.95
    )
    
    formato_comercial = np.random.choice(
        ["Tienda pequeña", "Tienda mediana", "Gran superficie (>2500 m²)", "Hipermercado", "Mall/Centro Comercial", "Local independiente"],
        p=[0.3, 0.25, 0.15, 0.1, 0.15, 0.05]
    )
    
    numero_locales = np.random.choice(
        ["1", "2-5", "6-20", "21-100", ">100"],
        p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )

    # === Convertir a valores numéricos para reglas ===
    if numero_pisos == "1":
        pisos_num = 1
    elif numero_pisos == "2":
        pisos_num = 2
    elif numero_pisos == "3":
        pisos_num = 3
    else:
        pisos_num = 5  # valor representativo para >3
    
    if area_total == "<300":
        area_total_num = 200
    elif area_total == "300-750":
        area_total_num = 525
    elif area_total == "750-2000":
        area_total_num = 1375
    elif area_total == "2000-10000":
        area_total_num = 6000
    else:
        area_total_num = 15000
    
    # === Aplicar reglas NORMATIVAS (en orden de prioridad) ===
    if comercializa_explosivos or any(p in tipo_productos.lower() for p in palabras_7_6):
        subfuncion = "7.6"
    elif ("Módulo" in tipo_establecimiento or "Stand" in tipo_establecimiento or "Puesto" in tipo_establecimiento) and tipo_licencia == "Corporativa (galería/mercado)":
        subfuncion = "7.2"
    elif tipo_establecimiento in ["Mercado Minorista", "Mercado Mayorista", "Supermercado", "Tienda por Departamentos", "Galería Comercial", "Centro Comercial", "Complejo Comercial"]:
        subfuncion = "7.5"  # ← Esta condición debe ir ANTES que 7.3
    elif "Áreas comunes" in uso_edificacion or "Mixto" in uso_edificacion:
        subfuncion = "7.4"
    elif pisos_num > 3 or area_total_num > 750:
        subfuncion = "7.3"  # ← Esta debe ir DESPUÉS de 7.5
    else:
        subfuncion = "7.1"

    # === AÑADIR RUIDO (5%) ===
    # if np.random.rand() < 0.01:
    #     subfuncion = np.random.choice(["7.1", "7.2", "7.3", "7.4", "7.5", "7.6"])

    # === CARACTERÍSTICAS ESCALABLES ===
    # 1. Cumple 7.1: ≤3 pisos y ≤750 m²
    cumple_7_1 = 1 if (pisos_num <= 3 and area_total_num <= 750) else 0
    
    # 2. Es 7.3: >3 pisos o >750 m²
    es_7_3 = 1 if (pisos_num > 3 or area_total_num > 750) else 0
    
    # 3. Es 7.2: módulo con licencia corporativa
    es_7_2 = 1 if (
        ("módulo" in tipo_establecimiento.lower() or "stand" in tipo_establecimiento.lower() or "puesto" in tipo_establecimiento.lower()) and
        tipo_licencia == "Corporativa (galería/mercado)"
    ) else 0
    
    # 4. Es 7.4: uso mixto o áreas comunes
    es_7_4 = 1 if ("mixto" in uso_edificacion.lower() or "áreas comunes" in uso_edificacion.lower()) else 0
    
    # 5. Es 7.5: establecimientos comerciales grandes
    es_7_5 = 1 if any(est in tipo_establecimiento for est in [
        "Mercado Minorista", "Mercado Mayorista", "Supermercado", "Tienda por Departamentos", 
        "Galería Comercial", "Centro Comercial", "Complejo Comercial"
    ]) else 0
    
    # 6. Es 7.6: productos peligrosos
    es_7_6 = 1 if comercializa_explosivos or any(p in tipo_productos.lower() for p in palabras_7_6) else 0
    
    # 7. Área total numérica
    area_total_num_final = area_total_num
    
    # 8. Número de pisos numérico
    pisos_num_final = pisos_num
    
    # 9. Área venta numérica
    area_venta_map = {"<200": 100, "200-500": 350, "500-1500": 1000, "1500-5000": 3250, ">5000": 7500}
    area_venta_num = area_venta_map.get(area_venta, 350)
    
    # 10. Número de locales numérico
    locales_map = {"1": 1, "2-5": 3, "6-20": 13, "21-100": 60, ">100": 150}
    locales_num = locales_map.get(numero_locales, 1)
    
    # 11. Modalidad operación (0=independiente, 1=módulo, 2=áreas comunes)
    if "independiente" in modalidad_operacion.lower():
        modalidad_num = 0
    elif "módulo" in modalidad_operacion.lower():
        modalidad_num = 1
    else:
        modalidad_num = 2

    data.append([
        cumple_7_1,
        es_7_3,
        es_7_2,
        es_7_4,
        es_7_5,
        es_7_6,
        area_total_num_final,
        pisos_num_final,
        area_venta_num,
        locales_num,
        modalidad_num,
        subfuncion
    ])

# === Guardar y entrenar ===
df = pd.DataFrame(data, columns=[
    "cumple_7_1", "es_7_3", "es_7_2", "es_7_4", "es_7_5", "es_7_6",
    "area_total", "numero_pisos", "area_venta", "numero_locales", "modalidad_operacion",
    "subfuncion"
])

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/dataset_comercio.csv", index=False)

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
joblib.dump(modelo, "models/rf_comercio.pkl")
print("✅ Modelo COMERCIO ESCALABLE guardado en models/rf_comercio.pkl")