import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"

import streamlit as st
import requests
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import torch
import numpy as np  # 🔥 FALTABA

# -----------------------
# 🤖 MODELO WHISPER (GPU/CPU)
# -----------------------

@st.cache_resource
def cargar_modelo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = whisper.load_model("medium").to(device)
    return modelo, device

model, device = cargar_modelo()

# -----------------------
# 🧠 LIMPIEZA BÁSICA
# -----------------------

def limpiar_texto_medico(texto):
    texto = texto.lower()

    reemplazos = {
        "disney": "disnea",
        "torasico": "torácico",
        "aspirinas": "aspirina",
        "100 miligramos": "100 mg",
        "cindre": "síndrome",
        "coronario abuso": "coronario agudo",
        "pacente": "paciente",
        "pocente": "paciente"
    }

    for k, v in reemplazos.items():
        texto = texto.replace(k, v)

    return texto.strip()

# -----------------------
# 🎙️ AUDIO (CON BUFFER REAL)
# -----------------------

def grabar_y_transcribir_stream(buffer_audio, duracion=3, fs=16000):

    audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()

    # 🔥 concatenar audio anterior + nuevo
    if buffer_audio is not None:
        audio = np.concatenate((buffer_audio, audio))

    write("audio_temp.wav", fs, audio)

    resultado = model.transcribe(
        "audio_temp.wav",
        language="es",
        temperature=0,
        beam_size=5,
        fp16=(device == "cuda"),
        initial_prompt="Nota médica: paciente con síntomas, medicamentos y diagnóstico."
    )

    texto = resultado["text"]
    texto = limpiar_texto_medico(texto)

    return texto, audio

# -----------------------
# 🧠 ESTADO GLOBAL
# -----------------------

if "texto" not in st.session_state:
    st.session_state.texto = ""

if "texto_acumulado" not in st.session_state:
    st.session_state.texto_acumulado = ""

if "dictando" not in st.session_state:
    st.session_state.dictando = False

if "resultado_api" not in st.session_state:
    st.session_state.resultado_api = None

if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = None  # 🔥 CLAVE

# -----------------------
# 🧠 UI
# -----------------------

st.title("🧠 Asistente Clínico Inteligente")

opcion = st.radio("Tipo de entrada:", ["Texto", "Voz"])

# -----------------------
# ✍️ MODO TEXTO
# -----------------------

if opcion == "Texto":
    st.session_state.texto = st.text_area(
        "📄 Nota clínica",
        value=st.session_state.texto,
        height=150
    )

# -----------------------
# 🎙️ MODO VOZ
# -----------------------

elif opcion == "Voz":

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🟢 Iniciar"):
            st.session_state.dictando = True

    with col2:
        if st.button("🔴 Detener"):
            st.session_state.dictando = False

    with col3:
        if st.button("🔄 Regrabar"):
            st.session_state.texto_acumulado = ""
            st.session_state.audio_buffer = None  # 🔥 RESET REAL
            st.session_state.dictando = True
            st.rerun()

    # 🎙️ ESTADO
    if st.session_state.dictando:
        st.success("🎙️ Grabando... habla ahora")

        texto_nuevo, nuevo_buffer = grabar_y_transcribir_stream(
            st.session_state.audio_buffer
        )

        st.session_state.audio_buffer = nuevo_buffer  # 🔥 guardar buffer

        if texto_nuevo.strip():
            if texto_nuevo not in st.session_state.texto_acumulado:
                st.session_state.texto_acumulado = texto_nuevo  # 🔥 mejor overwrite que concatenar

        st.rerun()

    else:
        st.warning("⏹️ Dictado detenido")

    # 📄 TEXTO EN TIEMPO REAL
    st.text_area(
        "📄 Texto en tiempo real",
        st.session_state.texto_acumulado,
        height=200
    )

    st.session_state.texto = st.session_state.texto_acumulado

# -----------------------
# 📄 MOSTRAR TEXTO FINAL
# -----------------------

if st.session_state.texto:
    st.subheader("📄 Nota clínica detectada")
    st.write(st.session_state.texto)

# -----------------------
# 🚀 PROCESAR
# -----------------------

if st.button("🧠 Procesar nota clínica"):

    if not st.session_state.texto.strip():
        st.warning("⚠️ Ingresa o graba una nota clínica primero")

    else:
        with st.spinner("🔍 Analizando..."):

            try:
                response = requests.post(
                    "http://127.0.0.1:8000/procesar",
                    params={"texto": st.session_state.texto}
                )

                if response.status_code == 200:
                    st.session_state.resultado_api = response.json()
                else:
                    st.error(f"❌ Error API: {response.status_code}")
                    st.text(response.text)

            except Exception as e:
                st.error("❌ No se pudo conectar con la API")
                st.text(str(e))

# -----------------------
# 📊 RESULTADOS
# -----------------------

if st.session_state.resultado_api:

    resultado = st.session_state.resultado_api

    st.success("✅ Análisis completado")

    st.subheader("🧹 Texto corregido")
    st.write(resultado.get("texto_corregido", ""))

    st.subheader("📊 Entidades detectadas")
    st.json(resultado.get("datos", {}))

    st.subheader("🩺 Evaluación clínica")
    st.json(resultado.get("evaluacion_clinica", {}))

    st.subheader("🤖 Predicción IA")
    st.json(resultado.get("ia_prediccion", {}))

# -----------------------
# 🧹 LIMPIAR
# -----------------------

if st.button("🗑️ Limpiar todo"):
    st.session_state.texto = ""
    st.session_state.texto_acumulado = ""
    st.session_state.dictando = False
    st.session_state.resultado_api = None
    st.session_state.audio_buffer = None
    st.rerun()