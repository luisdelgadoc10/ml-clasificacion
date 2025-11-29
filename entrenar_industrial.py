# entrenar_industrial_escalable.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

np.random.seed(42)
data = []

# Palabras clave para 5.3 (explosivos/pirotécnicos)
palabras_5_3 = {
    "explosivo", "pirotécnico", "municion", "fuegos artificiales", 
    "dinamita", "pólvora", "detonador", "cohetes", "artificios"
}

for _ in range(1600):
    # Generar datos realistas con valores nuevos
    tipo_proceso = np.random.choice(
        ["Manual/Artesanal", "Semi-mecanizado", "Mecanizado", "Automatizado", "Altamente Automatizado", "Artesanal digital", "Fabricación aditiva"],
        p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.03, 0.02]
    )
    tipo_maquinaria = np.random.choice(
        ["Herramientas Manuales", "Maquinaria Eléctrica Portátil", "Maquinaria Industrial Fija", "Línea de Producción", "Robots/CNC", "Impresora 3D", "Equipo especializado"],
        p=[0.25, 0.25, 0.2, 0.15, 0.1, 0.03, 0.02]
    )
    escala_produccion = np.random.choice(
        ["Unitaria/Por Pedido", "Pequeña Serie", "Mediana Serie", "Gran Serie", "Producción Continua"],
        p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )
    
    # Producto fabricado (con nuevos tipos)
    tipo_producto = np.random.choice(
        ["Artesanía/Manualidades", "Productos Industriales Generales", "Explosivos", "Pirotécnicos", "Municiones", "Materiales Relacionados Explosivos", "Componentes electrónicos", "Textiles"],
        p=[0.25, 0.3, 0.1, 0.1, 0.08, 0.07, 0.05, 0.05]
    )
    
    trabaja_materiales_explosivos = (
        "explosivo" in tipo_producto.lower() or 
        "pirotécnico" in tipo_producto.lower() or
        np.random.rand() > 0.9
    )
    
    nivel_peligrosidad = np.random.choice(
        ["Bajo (no inflamables)", "Medio (inflamables Clase IIIA)", "Alto (inflamables Clase I-II)", "Muy Alto (explosivos/reactivos)"],
        p=[0.4, 0.3, 0.2, 0.1]
    )
    
    area_produccion = np.random.choice(
        ["<50", "50-200", "200-1000", "1000-5000", ">5000"],
        p=[0.25, 0.3, 0.25, 0.15, 0.05]
    )
    
    numero_trabajadores = np.random.choice(
        ["1-5", "6-10", "11-50", "51-200", ">200"],
        p=[0.3, 0.25, 0.25, 0.15, 0.05]
    )
    
    tiene_area_comercializacion = (
        "artesanal" in tipo_proceso.lower() or 
        "manual" in tipo_maquinaria.lower() or
        np.random.rand() > 0.7
    )
    
    tipo_establecimiento = np.random.choice(
        ["Taller Artesanal", "Taller Industrial", "Planta Industrial", "Fábrica", "Fábrica de Explosivos", "Fábrica de Pirotécnicos", "Centro de fabricación", "Laboratorio de producción"],
        p=[0.25, 0.2, 0.2, 0.15, 0.08, 0.07, 0.03, 0.02]
    )

    # === Aplicar reglas NORMATIVAS ===
    if (trabaja_materiales_explosivos or 
        any(palabra in tipo_producto.lower() for palabra in ["explosivo", "pirotécnico", "municion"]) or
        "explosivo" in tipo_establecimiento.lower() or
        "pirotécnico" in tipo_establecimiento.lower() or
        "Muy Alto" in nivel_peligrosidad):
        subfuncion = "5.3"
    elif ("Manual/Artesanal" in tipo_proceso or 
          "Herramientas Manuales" in tipo_maquinaria or
          "Taller Artesanal" in tipo_establecimiento or
          area_produccion in ["<50", "50-200"] and numero_trabajadores in ["1-5", "6-10"]):
        subfuncion = "5.1"
    else:
        subfuncion = "5.2"

    # === AÑADIR RUIDO (5%) ===
    # if np.random.rand() < 0.01:
    #     subfuncion = np.random.choice(["5.1", "5.2", "5.3"])

    # === CARACTERÍSTICAS ESCALABLES ===
    # 1. es_artesanal: basado en palabras clave
    es_artesanal = 1 if (
        "manual" in tipo_proceso.lower() or
        "herramienta" in tipo_maquinaria.lower() or
        "artesanal" in tipo_establecimiento.lower() or
        "artesanía" in tipo_producto.lower()
    ) else 0
    
    # 2. es_explosivo: basado en palabras clave
    es_explosivo = 1 if (
        trabaja_materiales_explosivos or
        any(p in tipo_producto.lower() for p in ["explosivo", "pirotécnico", "municion", "fuegos"]) or
        "Muy Alto" in nivel_peligrosidad or
        any(p in tipo_establecimiento.lower() for p in ["explosivo", "pirotécnico"])
    ) else 0
    
    # 3. escala_produccion numérica
    escala_map = {
        "Unitaria/Por Pedido": 1,
        "Pequeña Serie": 2,
        "Mediana Serie": 3,
        "Gran Serie": 4,
        "Producción Continua": 5
    }
    escala_num = escala_map.get(escala_produccion, 2)
    
    # 4. area_produccion numérica
    area_map = {"<50": 30, "50-200": 125, "200-1000": 600, "1000-5000": 3000, ">5000": 7500}
    area_num = area_map.get(area_produccion, 125)
    
    # 5. numero_trabajadores numérico
    trab_map = {"1-5": 3, "6-10": 8, "11-50": 30, "51-200": 125, ">200": 300}
    trab_num = trab_map.get(numero_trabajadores, 8)
    
    # 6. nivel_peligrosidad numérico
    peligro_map = {
        "Bajo (no inflamables)": 1,
        "Medio (inflamables Clase IIIA)": 2,
        "Alto (inflamables Clase I-II)": 3,
        "Muy Alto (explosivos/reactivos)": 4
    }
    peligro_num = peligro_map.get(nivel_peligrosidad, 1)

    data.append([
        es_artesanal,
        es_explosivo,
        escala_num,
        area_num,
        trab_num,
        peligro_num,
        int(tiene_area_comercializacion),
        subfuncion
    ])

# === Guardar y entrenar ===
df = pd.DataFrame(data, columns=[
    "es_artesanal", "es_explosivo", "escala_produccion", 
    "area_produccion_m2", "numero_trabajadores", "nivel_peligrosidad_insumos",
    "tiene_area_comercializacion_integrada", "subfuncion"
])

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/dataset_industrial.csv", index=False)

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
joblib.dump(modelo, "models/rf_industrial.pkl")
print("✅ Modelo INDUSTRIAL ESCALABLE guardado en models/rf_industrial.pkl")