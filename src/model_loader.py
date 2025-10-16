# src/model_loader.py
import joblib
import os
import numpy as np
from typing import Dict, Tuple

# === Rutas de modelos ===
MODEL_SALUD_PATH = os.path.join("models", "rf_salud.pkl")
MODEL_ENCUENTRO_PATH = os.path.join("models", "rf_encuentro.pkl")
MODEL_HOSPEDAJE_PATH = os.path.join("models", "rf_hospedaje.pkl")
MODEL_EDUCACION_PATH = os.path.join("models", "rf_educacion.pkl")
MODEL_INDUSTRIAL_PATH = os.path.join("models", "rf_industrial.pkl")
MODEL_OFICINAS_PATH = os.path.join("models", "rf_oficinas.pkl")
MODEL_COMERCIO_PATH = os.path.join("models", "rf_comercio.pkl")
MODEL_ALMACEN_PATH = os.path.join("models", "rf_almacen.pkl")

# === Cargar modelos ===
if not os.path.exists(MODEL_SALUD_PATH):
    raise FileNotFoundError(f"❌ Modelo SALUD no encontrado: {MODEL_SALUD_PATH}")
if not os.path.exists(MODEL_ENCUENTRO_PATH):
    raise FileNotFoundError(f"❌ Modelo ENCUENTRO no encontrado: {MODEL_ENCUENTRO_PATH}")
if not os.path.exists(MODEL_HOSPEDAJE_PATH):
    raise FileNotFoundError(f"❌ Modelo HOSPEDAJE no encontrado: {MODEL_HOSPEDAJE_PATH}")
if not os.path.exists(MODEL_EDUCACION_PATH):
    raise FileNotFoundError(f"❌ Modelo EDUCACION no encontrado: {MODEL_EDUCACION_PATH}")
if not os.path.exists(MODEL_INDUSTRIAL_PATH):
    raise FileNotFoundError(f"❌ Modelo INDUSTRIAL no encontrado: {MODEL_INDUSTRIAL_PATH}")
if not os.path.exists(MODEL_OFICINAS_PATH):
    raise FileNotFoundError(f"❌ Modelo OFICINAS no encontrado: {MODEL_OFICINAS_PATH}")
if not os.path.exists(MODEL_COMERCIO_PATH):
    raise FileNotFoundError(f"❌ Modelo COMERCIO no encontrado: {MODEL_COMERCIO_PATH}")
if not os.path.exists(MODEL_ALMACEN_PATH):
    raise FileNotFoundError(f"❌ Modelo ALMACÉN no encontrado: {MODEL_ALMACEN_PATH}")

model_salud = joblib.load(MODEL_SALUD_PATH)
model_encuentro = joblib.load(MODEL_ENCUENTRO_PATH)
model_hospedaje = joblib.load(MODEL_HOSPEDAJE_PATH)
model_educacion = joblib.load(MODEL_EDUCACION_PATH)
model_industrial = joblib.load(MODEL_INDUSTRIAL_PATH)
model_oficinas = joblib.load(MODEL_OFICINAS_PATH)
model_comercio = joblib.load(MODEL_COMERCIO_PATH)
model_almacen = joblib.load(MODEL_ALMACEN_PATH)

# === FUNCIÓN SALUD ===
def preprocess_salud(data: Dict) -> np.ndarray:
    """Preprocesamiento robusto para SALUD."""
    nivel_map = {"Primer": 1, "Segundo": 2, "Tercer": 3}
    tipo_map = {
        "Puesto": 1, "Posta": 1,
        "Consultorio": 2, "Consultorio médico": 2,
        "Centro de salud": 3, "Centro médico": 3, "Policlínico": 3, "Centro médico especializado": 3,
        "Hospital general": 4,
        "Hospital especializado": 5,
        "Instituto": 6
    }
    camas_map = {"0": 0, "1-10": 1, "11-50": 2, ">50": 3}
    capacidad_map = {"Baja": 1, "Media": 2, "Alta": 3}
    especialidades_map = {"0": 0, "1-5": 1, ">5": 2}
    pisos_map = {"1": 1, "2": 2, ">3": 3}

    nivel = nivel_map.get(data["nivel_atencion"], 1)
    tipo = tipo_map.get(data["tipo_establecimiento"], 7)
    camas = camas_map.get(data["camas_internamiento"], 0)
    capacidad = capacidad_map.get(data["capacidad_atencion"], 2)
    esp = especialidades_map.get(data["num_especialidades"], 1)
    pisos = pisos_map.get(data["num_pisos"], 2)

    servicios_clave = {"Urgencias", "Laboratorio", "Farmacia", "Radiología", "UCI"}
    num_servicios = len(set(data["servicios_disponibles"]) & servicios_clave)

    features = [
        nivel, tipo, camas, int(data["usuarios_no_autosuficientes"]),
        capacidad, num_servicios, int(data["urgencias_24h"]),
        esp, pisos, float(data["area_construida"]),
        int(data["personal_medico_total"])
    ]
    return np.array(features).reshape(1, -1)

def predict_salud_with_confidence(data: Dict) -> Tuple[str, float]:
    X = preprocess_salud(data)
    probas = model_salud.predict_proba(X)[0]
    classes = model_salud.classes_
    max_idx = np.argmax(probas)
    return str(classes[max_idx]), float(probas[max_idx])

# === FUNCIÓN ENCUENTRO (CORREGIDA Y ESCALABLE) ===
def preprocess_encuentro(data: Dict) -> np.ndarray:
    """
    Preprocesamiento escalable para ENCUENTRO.
    Acepta cualquier tipo_actividad sin fallar.
    """
    # Lista normativa de usos que son 2.4
    usos_2_4 = {
        "discoteca", "casino", "tragamonedas", "teatro", "cine", "sala_concierto",
        "anfiteatro", "auditorio", "centro_convenciones", "club", "estadio",
        "plaza_toro", "coliseo", "hipodromo", "velodromo", "autodromo",
        "polideportivo", "parque_diversion", "zoologico", "templo", "iglesia"
    }
    
    horario_map = {"diurno": 1, "nocturno": 2, "mixto": 3}
    
    # --- Manejo seguro de tipo_actividad ---
    tipo_actividad = data["tipo_actividad"].lower().strip()
    es_2_4 = 1 if tipo_actividad in usos_2_4 else 0  # Nuevos tipos → 0
    
    # --- Manejo seguro de horario ---
    horario = horario_map.get(data["horario_funcionamiento"], 1)  # default: diurno

    features = [
        es_2_4,
        data["carga_ocupantes"],
        int(data["ubicado_en_sotano"]),
        data["num_pisos"],
        data["area_total_m2"],
        int(data["evento_recurrente"]),
        horario
    ]
    return np.array(features).reshape(1, -1)

def predict_encuentro_with_confidence(data: Dict) -> Tuple[str, float]:
    X = preprocess_encuentro(data)
    probas = model_encuentro.predict_proba(X)[0]
    classes = model_encuentro.classes_
    max_idx = np.argmax(probas)
    return str(classes[max_idx]), float(probas[max_idx])

# === HOSPEDAJE: preprocesamiento ESCALABLE ===
def preprocess_hospedaje(data: Dict) -> np.ndarray:
    """
    Preprocesamiento escalable para HOSPEDAJE.
    - Acepta CUALQUIER tipo_hospedaje sin fallar.
    - Solo 'ecolodge' y 'albergue' se marcan como especiales (tipo_especial=1).
    - Todos los demás (incluyendo nuevos) → tipo_especial=0.
    """
    # Tipos especiales mencionados en la normativa (3.1)
    tipos_especiales = {"ecolodge", "albergue"}
    
    # --- Manejo escalable: cualquier tipo_hospedaje es válido ---
    tipo_input = data["tipo_hospedaje"].lower().strip()
    tipo_especial = 1 if tipo_input in tipos_especiales else 0  # Nuevos tipos → 0
    
    # Simular área de estacionamiento (en producción, debería ser input)
    # Aquí asumimos que si hay estacionamiento en sótano, área = 600m² (suficiente para 3.4)
    area_estacionamiento = 600.0 if data["estacionamiento_en_sotano"] else 0.0

    features = [
        data["categoria_estrellas"],      # 0-5
        tipo_especial,                    # 1 si es ecolodge/albergue, 0 si no
        data["num_pisos"],                # int
        int(data["tiene_sotano"]),        # 0/1
        data["num_habitaciones"],         # int
        data["capacidad_ocupantes"],      # int
        int(data["uso_mixto"]),           # 0/1
        int(data["tiene_estacionamiento"]), # 0/1
        int(data["estacionamiento_en_sotano"]), # 0/1
        area_estacionamiento              # float (clave para 3.4)
    ]
    return np.array(features).reshape(1, -1)

def predict_hospedaje_with_confidence(data: Dict) -> Tuple[str, float]:
    X = preprocess_hospedaje(data)
    probas = model_hospedaje.predict_proba(X)[0]
    classes = model_hospedaje.classes_
    max_idx = np.argmax(probas)
    return str(classes[max_idx]), float(probas[max_idx])

# === FUNCIÓN EDUCACIÓN (ESCALABLE) ===
def preprocess_educacion(data: Dict) -> np.ndarray:
    """
    Preprocesamiento 100% escalable para EDUCACIÓN.
    - Acepta CUALQUIER valor en nivel_educativo y tipo_institucion.
    - Usa lógica normativa para derivar características clave.
    - Nunca falla con KeyError.
    """
    # 1. ¿Es educación básica?
    nivel_educativo_input = data["nivel_educativo"].lower().strip()
    niveles_basicos = {"inicial", "primaria", "secundaria"}
    es_basico = 1 if nivel_educativo_input in niveles_basicos else 0
    
    # 2. ¿Es educación superior?
    instituciones_superior = {
        "instituto", "escuela superior", "centro superior", 
        "universidad", "superior técnico", "superior universitario"
    }
    tipo_institucion_input = data["tipo_institucion"].lower().strip()
    es_superior = 1 if (
        tipo_institucion_input in instituciones_superior or 
        "superior" in nivel_educativo_input
    ) else 0
    
    # 3. Número de pisos numérico
    numero_pisos_str = data["numero_pisos"]
    if numero_pisos_str == ">10":
        num_pisos = 11
    elif numero_pisos_str == "6-10":
        num_pisos = 8
    else:
        try:
            num_pisos = int(numero_pisos_str)
        except (ValueError, TypeError):
            num_pisos = 3
    
    # 4. Área construida numérica
    area_str = data["area_construida_m2"]
    area_map = {
        "<500": 300,
        "500-1500": 1000,
        "1500-5000": 3000,
        "5000-15000": 10000,
        ">15000": 20000
    }
    area_num = area_map.get(area_str, 1000)
    
    # 5. Capacidad alumnos numérica
    cap_str = data["capacidad_alumnos"]
    cap_map = {
        "<100": 50,
        "100-300": 200,
        "300-800": 500,
        "800-2000": 1500,
        ">2000": 3000
    }
    cap_num = cap_map.get(cap_str, 200)
    
    # 6. Tipo de edificación
    tipo_edif_input = data["tipo_edificacion"].lower().strip()
    es_remoldeada = 1 if "remodelada" in tipo_edif_input or "acondicionada" in tipo_edif_input else 0

    features = [
        es_basico,
        es_superior,
        num_pisos,
        area_num,
        int(data["atiende_personas_discapacidad"]),
        cap_num,
        data["cantidad_aulas"],
        es_remoldeada
    ]
    return np.array(features).reshape(1, -1)

def predict_educacion_with_confidence(data: Dict) -> Tuple[str, float]:
    X = preprocess_educacion(data)
    probas = model_educacion.predict_proba(X)[0]
    classes = model_educacion.classes_
    max_idx = np.argmax(probas)
    return str(classes[max_idx]), float(probas[max_idx])

# === INDUSTRIAL: preprocesamiento ESCALABLE ===
def preprocess_industrial(data: Dict) -> np.ndarray:
    """
    Preprocesamiento 100% escalable para INDUSTRIAL.
    - Acepta CUALQUIER valor en las variables categóricas.
    - Usa lógica normativa con palabras clave.
    - Nunca falla con KeyError.
    """
    # --- 1. es_artesanal ---
    tipo_proceso = data["tipo_proceso_productivo"].lower()
    tipo_maquinaria = data["tipo_maquinaria_principal"].lower()
    tipo_establecimiento = data["tipo_establecimiento"].lower()
    tipo_producto = data["tipo_producto_fabricado"].lower()
    
    es_artesanal = 1 if (
        "manual" in tipo_proceso or
        "herramienta" in tipo_maquinaria or
        "artesanal" in tipo_establecimiento or
        "artesanía" in tipo_producto
    ) else 0
    
    # --- 2. es_explosivo ---
    trabaja_explosivos = data["trabaja_materiales_explosivos"]
    nivel_peligrosidad = data["nivel_peligrosidad_insumos"].lower()
    
    es_explosivo = 1 if (
        trabaja_explosivos or
        any(p in tipo_producto for p in ["explosivo", "pirotécnico", "municion", "fuegos", "pólvora"]) or
        "muy alto" in nivel_peligrosidad or
        any(p in tipo_establecimiento for p in ["explosivo", "pirotécnico"])
    ) else 0
    
    # --- 3. escala_produccion numérica ---
    escala_map = {
        "unitaria/por pedido": 1,
        "pequeña serie": 2,
        "mediana serie": 3,
        "gran serie": 4,
        "producción continua": 5
    }
    escala_input = data["escala_produccion"].lower()
    escala_num = escala_map.get(escala_input, 2)  # default: Pequeña Serie
    
    # --- 4. area_produccion numérica ---
    area_map = {"<50": 30, "50-200": 125, "200-1000": 600, "1000-5000": 3000, ">5000": 7500}
    area_input = data["area_produccion_m2"]
    area_num = area_map.get(area_input, 125)
    
    # --- 5. numero_trabajadores numérico ---
    trab_map = {"1-5": 3, "6-10": 8, "11-50": 30, "51-200": 125, ">200": 300}
    trab_input = data["numero_trabajadores"]
    trab_num = trab_map.get(trab_input, 8)
    
    # --- 6. nivel_peligrosidad numérico ---
    peligro_map = {
        "bajo (no inflamables)": 1,
        "medio (inflamables clase iiia)": 2,
        "alto (inflamables clase i-ii)": 3,
        "muy alto (explosivos/reactivos)": 4
    }
    peligro_input = data["nivel_peligrosidad_insumos"].lower()
    peligro_num = peligro_map.get(peligro_input, 1)
    
    # --- 7. area comercialización ---
    tiene_comercializacion = int(data["tiene_area_comercializacion_integrada"])

    features = [
        es_artesanal,
        es_explosivo,
        escala_num,
        area_num,
        trab_num,
        peligro_num,
        tiene_comercializacion
    ]
    return np.array(features).reshape(1, -1)

def predict_industrial_with_confidence(data: Dict) -> Tuple[str, float]:
    X = preprocess_industrial(data)
    probas = model_industrial.predict_proba(X)[0]
    classes = model_industrial.classes_
    max_idx = np.argmax(probas)
    return str(classes[max_idx]), float(probas[max_idx])

# === OFICINAS: preprocesamiento ESCALABLE ===
def preprocess_oficinas(data: Dict) -> np.ndarray:
    """
    Preprocesamiento 100% escalable para OFICINAS ADMINISTRATIVAS.
    - Acepta CUALQUIER valor en variables categóricas.
    - Usa lógica normativa con reglas claras.
    - Nunca falla con KeyError.
    """
    # === 1. Convertir área por piso a numérico ===
    area_por_piso_str = data["area_techada_por_piso_m2"]
    area_por_piso_map = {
        "<200": 100,
        "200-400": 300,
        "400-560": 480,
        "560-1000": 780,
        "1000-2500": 1750,
        ">2500": 3000
    }
    area_por_piso_num = area_por_piso_map.get(area_por_piso_str, 300)
    
    # === 2. Convertir número de pisos a numérico ===
    pisos_str = data["numero_pisos_edificacion"]
    if pisos_str == "1":
        pisos_num = 1
    elif pisos_str == "2":
        pisos_num = 2
    elif pisos_str == "3":
        pisos_num = 3
    elif pisos_str == "4":
        pisos_num = 4
    else:
        pisos_num = 6  # valor representativo para >4
    
    # === 3. Características derivadas ===
    # Cumple 6.1: ≤4 pisos y ≤560 m² por piso
    cumple_6_1 = 1 if (pisos_num <= 4 and area_por_piso_num <= 560) else 0
    
    # Es 6.5: >560 m² por piso
    es_6_5 = 1 if area_por_piso_num > 560 else 0
    
    # Conformidad reciente (≤5 años en 2025)
    año_actual = 2025
    antigüedad = año_actual - data["año_conformidad_obra"]
    conformidad_reciente = 1 if (data["tiene_conformidad_obra_vigente"] and antigüedad <= 5) else 0
    
    # Uso compartido
    uso_compartido = 1 if "compartido" in data["tipo_ocupacion_edificio"].lower() else 0
    
    # ITSE vigente
    itse_vigente = 1 if data["areas_comunes_tienen_itse_vigente"].lower() == "sí" else 0
    
    # Área total numérica
    area_total_map = {"<500": 300, "500-2000": 1250, "2000-5000": 3500, "5000-15000": 10000, ">15000": 20000}
    area_total_num = area_total_map.get(data["area_techada_total_m2"], 1250)
    
    # Año conformidad
    año_conformidad = data["año_conformidad_obra"]
    
    # Ha remodelado
    ha_remodelado = int(data["ha_tenido_remodelaciones_ampliaciones"])

    features = [
        cumple_6_1,
        es_6_5,
        conformidad_reciente,
        uso_compartido,
        itse_vigente,
        area_por_piso_num,
        pisos_num,
        area_total_num,
        año_conformidad,
        ha_remodelado
    ]
    return np.array(features).reshape(1, -1)

def predict_oficinas_with_confidence(data: Dict) -> Tuple[str, float]:
    X = preprocess_oficinas(data)
    probas = model_oficinas.predict_proba(X)[0]
    classes = model_oficinas.classes_
    max_idx = np.argmax(probas)
    return str(classes[max_idx]), float(probas[max_idx])

# === COMERCIO: preprocesamiento ESCALABLE ===
def preprocess_comercio(data: Dict) -> np.ndarray:
    """
    Preprocesamiento 100% escalable para COMERCIO.
    - Acepta CUALQUIER valor en variables categóricas.
    - Usa lógica normativa con palabras clave.
    - Nunca falla con KeyError.
    """
    # === 1. Convertir área total a numérico ===
    area_total_str = data["area_techada_total_m2"]
    area_total_map = {"<300": 200, "300-750": 525, "750-2000": 1375, "2000-10000": 6000, ">10000": 15000}
    area_total_num = area_total_map.get(area_total_str, 525)
    
    # === 2. Convertir número de pisos a numérico ===
    pisos_str = data["numero_pisos_edificacion"]
    if pisos_str == "1":
        pisos_num = 1
    elif pisos_str == "2":
        pisos_num = 2
    elif pisos_str == "3":
        pisos_num = 3
    else:
        pisos_num = 5  # valor representativo para >3
    
    # === 3. Características derivadas ===
    # Cumple 7.1: ≤3 pisos y ≤750 m²
    cumple_7_1 = 1 if (pisos_num <= 3 and area_total_num <= 750) else 0
    
    # Es 7.3: >3 pisos o >750 m²
    es_7_3 = 1 if (pisos_num > 3 or area_total_num > 750) else 0
    
    # Es 7.2: módulo con licencia corporativa
    tipo_establecimiento = data["tipo_establecimiento_comercial"].lower()
    tipo_licencia = data["tipo_licencia_funcionamiento"]
    es_7_2 = 1 if (
        ("módulo" in tipo_establecimiento or "stand" in tipo_establecimiento or "puesto" in tipo_establecimiento) and
        tipo_licencia == "Corporativa (galería/mercado)"
    ) else 0
    
    # Es 7.4: uso mixto o áreas comunes
    uso_edificacion = data["uso_edificacion"].lower()
    es_7_4 = 1 if ("mixto" in uso_edificacion or "áreas comunes" in uso_edificacion) else 0
    
    # Es 7.5: establecimientos comerciales grandes
    establecimientos_7_5 = {
        "mercado minorista", "mercado mayorista", "supermercado", "tienda por departamentos",
        "galería comercial", "centro comercial", "complejo comercial"
    }
    es_7_5 = 1 if any(est in tipo_establecimiento for est in establecimientos_7_5) else 0
    
    # Es 7.6: productos peligrosos
    comercializa_explosivos = data["comercializa_productos_explosivos_pirotecnicos"]
    tipo_productos = data["tipo_productos_peligrosos"].lower()
    palabras_7_6 = {"explosivo", "pirotécnico", "municion", "fuegos", "pólvora"}
    es_7_6 = 1 if (
        comercializa_explosivos or 
        any(p in tipo_productos for p in palabras_7_6)
    ) else 0
    
    # Área venta numérica
    area_venta_map = {"<200": 100, "200-500": 350, "500-1500": 1000, "1500-5000": 3250, ">5000": 7500}
    area_venta_num = area_venta_map.get(data["area_venta_m2"], 350)
    
    # Número de locales numérico
    locales_map = {"1": 1, "2-5": 3, "6-20": 13, "21-100": 60, ">100": 150}
    locales_num = locales_map.get(data["numero_locales_comerciales_edificio"], 1)
    
    # Modalidad operación numérica
    modalidad = data["modalidad_operacion"].lower()
    if "independiente" in modalidad:
        modalidad_num = 0
    elif "módulo" in modalidad:
        modalidad_num = 1
    else:
        modalidad_num = 2

    features = [
        cumple_7_1,
        es_7_3,
        es_7_2,
        es_7_4,
        es_7_5,
        es_7_6,
        area_total_num,
        pisos_num,
        area_venta_num,
        locales_num,
        modalidad_num
    ]
    return np.array(features).reshape(1, -1)

def predict_comercio_with_confidence(data: Dict) -> Tuple[str, float]:
    X = preprocess_comercio(data)
    probas = model_comercio.predict_proba(X)[0]
    classes = model_comercio.classes_
    max_idx = np.argmax(probas)
    return str(classes[max_idx]), float(probas[max_idx])

# === ALMACEN: preprocesamiento ESCALABLE ===
def preprocess_almacen(data: Dict) -> np.ndarray:
    """
    Preprocesamiento 100% escalable para ALMACÉN.
    - Acepta CUALQUIER valor en variables categóricas.
    - Usa lógica normativa con palabras clave.
    - Nunca falla con KeyError.
    """
    # === 1. Es 8.3: productos peligrosos ===
    almacena_explosivos = data["almacena_productos_explosivos_pirotecnicos"]
    tipo_productos = data["tipo_productos_almacenados"].lower()
    palabras_8_3 = {"explosivo", "pirotécnico", "municion", "fuegos", "pólvora"}
    es_8_3 = 1 if (
        almacena_explosivos or 
        any(p in tipo_productos for p in palabras_8_3)
    ) else 0
    
    # === 2. Es 8.1: no techado ===
    tipo_cobertura = data["tipo_cobertura"]
    porcentaje_techado = data["porcentaje_area_techada"]
    es_8_1 = 1 if (tipo_cobertura == "No Techado" or porcentaje_techado == "0%") else 0
    
    # === 3. Porcentaje techado numérico ===
    porcentaje_map = {"0%": 0, "1-25%": 15, "26-50%": 37, "51-75%": 62, "76-99%": 87, "100%": 100}
    porcentaje_num = porcentaje_map.get(porcentaje_techado, 50)
    
    # === 4. Tipo cobertura numérico ===
    cobertura_map = {
        "no techado": 0,
        "parcialmente techado": 1,
        "totalmente techado": 2,
        "cerrado y techado": 3
    }
    cobertura_num = cobertura_map.get(tipo_cobertura.lower(), 1)
    
    # === 5. Tipo cerramiento numérico ===
    cerramiento_map = {
        "abierto": 0,
        "semi-abierto (muros parciales)": 1,
        "cerrado (muros completos)": 2,
        "con climatización": 3
    }
    cerramiento_num = cerramiento_map.get(data["tipo_cerramiento"].lower(), 1)
    
    # === 6. Nivel NFPA numérico ===
    nfpa_map = {
        "0 (mínimo)": 0,
        "1 (ligero)": 1,
        "2 (moderado)": 2,
        "3 (serio)": 3,
        "4 (severo)": 4
    }
    nfpa_num = nfpa_map.get(data["nivel_peligrosidad_nfpa"], 0)
    
    # === 7. Tiene áreas administrativas ===
    tiene_areas_admin = int(data["tiene_areas_administrativas_techadas"])
    
    # === 8. Área administrativa numérica ===
    area_admin_map = {"0": 0, "1-50": 30, "51-200": 125, "201-500": 350, ">500": 750}
    area_admin_num = area_admin_map.get(data["area_administrativa_servicios_m2"], 30)
    
    # === 9. Es estacionamiento ===
    uso_principal = data["uso_principal"].lower()
    tipo_establecimiento = data["tipo_establecimiento"].lower()
    es_estacionamiento = 1 if ("estacionamiento" in uso_principal or "vehicular" in tipo_establecimiento) else 0

    features = [
        es_8_3,
        es_8_1,
        porcentaje_num,
        cobertura_num,
        cerramiento_num,
        nfpa_num,
        tiene_areas_admin,
        area_admin_num,
        es_estacionamiento
    ]
    return np.array(features).reshape(1, -1)

def predict_almacen_with_confidence(data: Dict) -> Tuple[str, float]:
    X = preprocess_almacen(data)
    probas = model_almacen.predict_proba(X)[0]
    classes = model_almacen.classes_
    max_idx = np.argmax(probas)
    return str(classes[max_idx]), float(probas[max_idx])