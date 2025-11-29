# entrenar_encuentro.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

np.random.seed(42)
data = []

# Usos que DEFINITIVAMENTE son 2.4 (normativa)
usos_2_4 = {
    "discoteca", "casino", "tragamonedas", "teatro", "cine", "sala_concierto",
    "anfiteatro", "auditorio", "centro_convenciones", "club", "estadio",
    "plaza_toro", "coliseo", "hipodromo", "velodromo", "autodromo",
    "polideportivo", "parque_diversion", "zoologico", "templo", "iglesia"
}

# También incluimos algunos "Otros" conocidos
otros_conocidos = ["salon_eventos", "auditorio_escolar", "comunidad", "salon_comunal", "gimnasio", "biblioteca"]

for _ in range(1800):
    # Decidir si es uso 2.4, "Otros conocido", o "Nuevo tipo"
    r = np.random.rand()
    if r < 0.55:
        tipo_actividad = np.random.choice(list(usos_2_4))
        es_2_4 = True
        tipo_actividad_num = 1  # 1 = 2.4
    elif r < 0.85:
        tipo_actividad = np.random.choice(otros_conocidos)
        es_2_4 = False
        tipo_actividad_num = 0  # 0 = No 2.4
    else:
        # ¡Nuevo tipo no visto! (simulamos nombres futuros)
        tipo_actividad = np.random.choice([
            "arena_virtual", "centro_espiritual", "lounge_ejecutivo",
            "parque_tecnologico", "museo_interactivo", "centro_cultural"
        ])
        es_2_4 = False
        tipo_actividad_num = 0  # Por defecto, no es 2.4

    carga_ocupantes = np.random.randint(10, 600)
    ubicado_en_sotano = np.random.choice([True, False], p=[0.15, 0.85])
    num_pisos = np.random.randint(1, 7)
    area_total_m2 = np.random.uniform(50, 12000)
    evento_recurrente = es_2_4 or np.random.rand() > 0.4
    horario_funcionamiento = np.random.choice(["diurno", "nocturno", "mixto"])

    # Aplicar reglas NORMATIVAS (prioridad fija)
    if ubicado_en_sotano:
        subfuncion = "2.3"
    elif carga_ocupantes <= 50:
        subfuncion = "2.1"
    elif es_2_4:
        subfuncion = "2.4"
    else:
        subfuncion = "2.2"

    # Mapear horario
    horario_map = {"diurno": 1, "nocturno": 2, "mixto": 3}
    horario = horario_map[horario_funcionamiento]

    data.append([
        tipo_actividad_num,  # 1=2.4, 0=No 2.4
        carga_ocupantes,
        int(ubicado_en_sotano),
        num_pisos,
        area_total_m2,
        int(evento_recurrente),
        horario,
        subfuncion
    ])

# Crear y guardar
df = pd.DataFrame(data, columns=[
    "es_2_4", "carga_ocupantes", "ubicado_en_sotano",
    "num_pisos", "area_total_m2", "evento_recurrente",
    "horario_funcionamiento", "subfuncion"
])

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/dataset_encuentro.csv", index=False)

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
joblib.dump(modelo, "models/rf_encuentro.pkl")
print("✅ Modelo ENCUENTRO escalable guardado.")