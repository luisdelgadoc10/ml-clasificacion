from pydantic import BaseModel
from typing import List

# ===== FUNCION SALUD =====
class FuncionSaludInput(BaseModel):
    nivel_atencion: str  # "Primer", "Segundo", "Tercer"
    tipo_establecimiento: str  # "Puesto", "Consultorio médico", etc.
    camas_internamiento: str  # "0", "1-10", "11-50", ">50"
    usuarios_no_autosuficientes: bool
    capacidad_atencion: str  # "Baja", "Media", "Alta"
    servicios_disponibles: List[str]  # ej: ["Urgencias", "Laboratorio"]
    urgencias_24h: bool
    num_especialidades: str  # "0", "1-5", ">5"
    num_pisos: str  # "1", "2", ">3"
    area_construida: float
    personal_medico_total: int

# ===== FUNCION ENCUENTRO =====
class FuncionEncuentroInput(BaseModel):
    tipo_actividad: str  # "salon_eventos", "discoteca", etc.
    carga_ocupantes: int
    ubicado_en_sotano: bool
    num_pisos: int
    area_total_m2: float
    evento_recurrente: bool
    horario_funcionamiento: str  # "diurno", "nocturno", "mixto"

# ===== FUNCION HOSPEDAJE =====
class FuncionHospedajeInput(BaseModel):
    categoria_estrellas: int  # 0-5
    tipo_hospedaje: str       # "hotel", "hostal", etc.
    num_pisos: int
    tiene_sotano: bool
    num_habitaciones: int
    capacidad_ocupantes: int
    uso_mixto: bool
    tiene_estacionamiento: bool
    estacionamiento_en_sotano: bool

# ===== FUNCION EDUCACION =====
class FuncionEducacionInput(BaseModel):
    nivel_educativo: str  # "Inicial", "Primaria", etc.
    tipo_institucion: str  # "CEBE", "Colegio Regular", etc.
    numero_pisos: str  # "1", "2", "3", "4", "5", "6-10", ">10"
    area_construida_m2: str  # "<500", "500-1500", etc.
    atiende_personas_discapacidad: bool
    capacidad_alumnos: str  # "<100", "100-300", etc.
    cantidad_aulas: int
    tipo_edificacion: str  # "Construida como Educativa" / "Remodelada/Acondicionada para Educación"


# ===== FUNCION INDUSTRIAL =====
class FuncionIndustrialInput(BaseModel):
    tipo_proceso_productivo: str  # "Manual/Artesanal", etc.
    tipo_maquinaria_principal: str  # "Herramientas Manuales", etc.
    escala_produccion: str  # "Unitaria/Por Pedido", etc.
    trabaja_materiales_explosivos: bool
    tipo_producto_fabricado: str  # "Artesanía/Manualidades", etc.
    nivel_peligrosidad_insumos: str  # "Bajo", "Medio", etc.
    area_produccion_m2: str  # "<50", "50-200", etc.
    numero_trabajadores: str  # "1-5", "6-10", etc.
    tiene_area_comercializacion_integrada: bool
    tipo_establecimiento: str  # "Taller Artesanal", etc.


# ===== FUNCION OFICINAS ADMINISTRATIVAS =====
class FuncionOficinasInput(BaseModel):
    numero_pisos_edificacion: str  # "1", "2", "3", "4", "5-10", "11-20", ">20"
    area_techada_por_piso_m2: str  # "<200", "200-400", etc.
    area_techada_total_m2: str     # "<500", "500-2000", etc.
    año_conformidad_obra: int      # 2020, 2021, etc.
    antigüedad_conformidad_años: str  # "0-1", "2-3", etc.
    tiene_conformidad_obra_vigente: bool
    tipo_conformidad: str          # "Obra Nueva", "Remodelación", etc.
    tipo_ocupacion_edificio: str   # "Uso Exclusivo", "Uso Compartido"
    areas_comunes_tienen_itse_vigente: str  # "Sí", "No", "No Aplica"
    piso_ubicacion_establecimiento: str     # "PB", "1", "2", etc.
    uso_diseño_original: str       # "Oficinas desde origen", "Adaptado a oficinas"
    ha_tenido_remodelaciones_ampliaciones: bool

# ===== FUNCION COMERCIO =====
class FuncionComercioInput(BaseModel):
    numero_pisos_edificacion: str  # "1", "2", "3", "4", "5-10", ">10"
    area_techada_total_m2: str     # "<300", "300-750", etc.
    area_venta_m2: str             # "<200", "200-500", etc.
    tipo_establecimiento_comercial: str  # "Tienda Individual", etc.
    modalidad_operacion: str       # "Independiente", etc.
    uso_edificacion: str           # "Comercial Exclusivo", etc.
    tipo_licencia_funcionamiento: str  # "Individual", etc.
    edificio_tiene_licencia_corporativa: str  # "Sí", "No", "No Aplica"
    comercializa_productos_explosivos_pirotecnicos: bool
    tipo_productos_peligrosos: str  # "Ninguno", "Explosivos", etc.
    formato_comercial: str         # "Tienda pequeña", etc.
    numero_locales_comerciales_edificio: str  # "1", "2-5", etc.

# ===== FUNCION ALMACEN =====
class FuncionAlmacenInput(BaseModel):
    tipo_cobertura: str  # "No Techado", "Parcialmente Techado", etc.
    porcentaje_area_techada: str  # "0%", "1-25%", etc.
    tipo_cerramiento: str  # "Abierto", "Semi-abierto", etc.
    tipo_establecimiento: str  # "Almacén General", "Depósito", etc.
    uso_principal: str  # "Almacenamiento de mercancías", etc.
    almacena_productos_explosivos_pirotecnicos: bool
    tipo_productos_almacenados: str  # "Ninguno (vacío/vehículos)", etc.
    nivel_peligrosidad_nfpa: str  # "0 (mínimo)", "1 (ligero)", etc.
    tiene_areas_administrativas_techadas: bool
    area_administrativa_servicios_m2: str  # "0", "1-50", etc.
