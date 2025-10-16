# api/main.py
import time
from fastapi import FastAPI
from src.schemas import (
    FuncionSaludInput,
    FuncionEncuentroInput,
    FuncionHospedajeInput,
    FuncionEducacionInput,
    FuncionIndustrialInput,
    FuncionOficinasInput,
    FuncionComercioInput,
    FuncionAlmacenInput
)
from src.model_loader import (
    predict_salud_with_confidence,
    predict_encuentro_with_confidence,
    predict_hospedaje_with_confidence,
    predict_educacion_with_confidence,
    predict_industrial_with_confidence,
    predict_oficinas_with_confidence,
    predict_comercio_with_confidence,
    predict_almacen_with_confidence
) 


app = FastAPI(
    title="ML Matriz de Riesgos - Clasificación de Funciones",
    description="Devuelve subfunción, confianza (%) y tiempo de predicción (ms)",
    version="1.2"
)

@app.post("/funcion-salud")
def clasificar_funcion_salud(entrada: FuncionSaludInput):
    """
    Recibe datos del establecimiento y devuelve:
    - Subfunción de salud
    - Confianza (%)
    - Tiempo de predicción (milisegundos)
    """
    # Medir tiempo de predicción
    start_time = time.perf_counter()
    resultado, confianza = predict_salud_with_confidence(entrada.dict())
    end_time = time.perf_counter()
    
    tiempo_ms = round((end_time - start_time) * 1000, 2)  # Convertir a milisegundos

    return {
        "subfuncion_salud": resultado,
        "confianza": round(confianza * 100, 2),
        "tiempo_ms": tiempo_ms
    }

# === Endpoint ENCUENTRO ===
@app.post("/funcion-encuentro")
def clasificar_encuentro(entrada: FuncionEncuentroInput):
    start = time.perf_counter()
    resultado, confianza = predict_encuentro_with_confidence(entrada.dict())
    tiempo_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "subfuncion_encuentro": resultado,
        "confianza": round(confianza * 100, 2),
        "tiempo_ms": tiempo_ms
    }

# === Endpoint HOSPEDAJE ===
@app.post("/funcion-hospedaje")
def clasificar_hospedaje(entrada: FuncionHospedajeInput):
    start = time.perf_counter()
    resultado, confianza = predict_hospedaje_with_confidence(entrada.dict())
    tiempo_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "subfuncion_hospedaje": resultado,
        "confianza": round(confianza * 100, 2),
        "tiempo_ms": tiempo_ms
    }

# === Endpoint EDUCACION ===
@app.post("/funcion-educacion")
def clasificar_educacion(entrada: FuncionEducacionInput):
    start = time.perf_counter()
    resultado, confianza = predict_educacion_with_confidence(entrada.dict())
    tiempo_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "subfuncion_educacion": resultado,
        "confianza": round(confianza * 100, 2),
        "tiempo_ms": tiempo_ms
    }

# === Endpoint INDUSTRIAL ===
@app.post("/funcion-industrial")
def clasificar_industrial(entrada: FuncionIndustrialInput):
    start = time.perf_counter()
    resultado, confianza = predict_industrial_with_confidence(entrada.dict())
    tiempo_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "subfuncion_industrial": resultado,
        "confianza": round(confianza * 100, 2),
        "tiempo_ms": tiempo_ms
    }

# === Endpoint OFICINAS ADMINISTRATIVAS ===
@app.post("/funcion-oficinas")
def clasificar_oficinas(entrada: FuncionOficinasInput):
    start = time.perf_counter()
    resultado, confianza = predict_oficinas_with_confidence(entrada.dict())
    tiempo_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "subfuncion_oficinas": resultado,
        "confianza": round(confianza * 100, 2),
        "tiempo_ms": tiempo_ms
    }

# === Endpoint COMERCIO ===
@app.post("/funcion-comercio")
def clasificar_comercio(entrada: FuncionComercioInput):
    start = time.perf_counter()
    resultado, confianza = predict_comercio_with_confidence(entrada.dict())
    tiempo_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "subfuncion_comercio": resultado,
        "confianza": round(confianza * 100, 2),
        "tiempo_ms": tiempo_ms
    }

# === Endpoint ALMACEN ===
@app.post("/funcion-almacen")
def clasificar_almacen(entrada: FuncionAlmacenInput):
    start = time.perf_counter()
    resultado, confianza = predict_almacen_with_confidence(entrada.dict())
    tiempo_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "subfuncion_almacen": resultado,
        "confianza": round(confianza * 100, 2),
        "tiempo_ms": tiempo_ms
    }