from fastapi import FastAPI
from app.clinical_corrector import CorrectorClinico
from app.nlp_transformers import MotorNLPTransformer as MotorNLP
from app.clinical_rules import MotorClinico
from app.ml_model import ModeloDiagnostico
import json
import re

app = FastAPI()

# ------------------------
# 🔧 INICIALIZACIÓN GLOBAL
# ------------------------

corrector = CorrectorClinico()
nlp = MotorNLP()
clinico = MotorClinico()

modelo_ml = ModeloDiagnostico()
modelo_ml.entrenar()

# ------------------------
# 📦 GUARDAR CASOS
# ------------------------

def guardar_caso(sintomas, diagnostico):
    try:
        with open("data/casos.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        data = []

    data.append({
        "sintomas": sintomas,
        "diagnostico": diagnostico
    })

    with open("data/casos.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ------------------------
# 🧠 VALIDACIÓN REAL (MEJORADA)
# ------------------------

def texto_valido(texto):

    texto = texto.lower()

    # mínimo de estructura clínica
    patrones = [
        "paciente",
        "dolor",
        "diagnóstico",
        "refiere",
        "presenta"
    ]

    score = sum(p in texto for p in patrones)

    # evita ruido extremo
    palabras = len(texto.split())

    return score >= 1 and palabras > 3

# ------------------------
# 🚦 SEMÁFORO CLÍNICO
# ------------------------

def generar_semaforo(entidades, diagnostico_predicho, evaluacion):

    sintomas = entidades.get("sintomas", [])
    alertas = evaluacion.get("alertas", [])

    if "dolor" in sintomas and "dificultad respiratoria" in sintomas:
        return {"nivel": "rojo", "mensaje": "🚨 Posible emergencia cardiopulmonar"}

    if "fiebre" in sintomas and "dificultad respiratoria" in sintomas:
        return {"nivel": "rojo", "mensaje": "🚨 Posible infección respiratoria grave"}

    if "⚠️ Diferencia entre diagnóstico y predicción" in alertas:
        return {"nivel": "amarillo", "mensaje": "Inconsistencia entre modelo e impresión clínica"}

    if len(entidades.get("diagnosticos", [])) == 0:
        return {"nivel": "amarillo", "mensaje": "Falta diagnóstico clínico"}

    if len(sintomas) == 0:
        return {"nivel": "amarillo", "mensaje": "Datos clínicos insuficientes"}

    return {"nivel": "verde", "mensaje": "Evaluación clínica consistente"}

# ------------------------
# 🚀 ENDPOINT PRINCIPAL
# ------------------------

@app.post("/procesar")
def procesar(texto: str):

    try:

        # ------------------------
        # 1. LIMPIEZA + NORMALIZACIÓN
        # ------------------------
        texto_corregido = corrector.corregir(texto)

        # ------------------------
        # 2. VALIDACIÓN POST-CORRECCIÓN
        # ------------------------
        if not texto_valido(texto_corregido):
            return {
                "estado": "error",
                "mensaje": "Texto no válido o demasiado ruidoso",
                "texto_recibido": texto
            }

        # ------------------------
        # 3. NLP
        # ------------------------
        entidades = nlp.extraer(texto_corregido)

        # ------------------------
        # 4. REGLAS CLÍNICAS
        # ------------------------
        evaluacion = clinico.evaluar(entidades)

        # ------------------------
        # 5. MODELO ML
        # ------------------------
        diagnostico_predicho = modelo_ml.predecir(entidades.get("sintomas", []))

        # ------------------------
        # 6. VALIDACIÓN CRUZADA
        # ------------------------
        alertas = evaluacion.get("alertas", [])

        if entidades.get("diagnosticos"):
            diagnostico_real = entidades["diagnosticos"][0]

            if diagnostico_real != diagnostico_predicho:
                alertas.append("⚠️ Diferencia entre diagnóstico y predicción")

        evaluacion["alertas"] = alertas

        # ------------------------
        # 7. SEMÁFORO CLÍNICO
        # ------------------------
        semaforo = generar_semaforo(entidades, diagnostico_predicho, evaluacion)

        # ------------------------
        # 8. APRENDIZAJE CONTROLADO
        # ------------------------
        if entidades.get("diagnosticos"):
            guardar_caso(entidades["sintomas"], entidades["diagnosticos"][0])
            modelo_ml.entrenar()

        # ------------------------
        # 9. RESPUESTA FINAL
        # ------------------------
        return {
            "estado": "ok",
            "texto_original": texto,
            "texto_corregido": texto_corregido,
            "datos": entidades,
            "evaluacion_clinica": evaluacion,
            "ia_prediccion": {
                "diagnostico_sugerido": diagnostico_predicho
            },
            "semaforo_clinico": semaforo
        }

    except Exception as e:
        return {
            "estado": "error",
            "mensaje": "Error interno en procesamiento",
            "detalle": str(e)
        }


@app.get("/")
def home():
    return {"mensaje": "API Asistente Clínico activa"}