from transformers import pipeline
import re
import json

# -----------------------
# 📚 DICCIONARIO MÉDICO
# -----------------------
with open("data/diccionario_medico.json", encoding="utf-8") as f:
    DICCIONARIO = json.load(f)

# -----------------------
# 🤖 NER (complemento)
# -----------------------
ner_pipeline = pipeline(
    "ner",
    model="Davlan/bert-base-multilingual-cased-ner-hrl",
    aggregation_strategy="simple"
)

# -----------------------
# 🧠 NLP CLÍNICO
# -----------------------
class MotorNLPTransformer:

    # -----------------------
    # 🔧 NORMALIZACIÓN
    # -----------------------
    def normalizar(self, texto):
        texto = texto.lower()

        reemplazos = {
            "torasico": "torácico",
            "toleracico": "torácico",
            "torasico opresivo": "dolor torácico opresivo",
            "disneaa": "disnea",
            "disnea": "dificultad respiratoria",
            "aspirinas": "aspirina",
            "100 miligramos": "100 mg",
            "miligramos": "mg",
            "pacente": "paciente",
            "pocente": "paciente",
            "diagnostico": "diagnóstico"
        }

        for k, v in reemplazos.items():
            texto = texto.replace(k, v)

        return texto

    # -----------------------
    # 📚 APLICAR DICCIONARIO (SIN ERROR)
    # -----------------------
    def aplicar_diccionario(self, texto):
        # No reemplaza listas, solo mantiene consistencia
        for categoria, lista in DICCIONARIO.items():
            for termino in lista:
                if termino in texto:
                    texto = texto.replace(termino, termino)
        return texto

    # -----------------------
    # 🚀 EXTRACCIÓN PRINCIPAL
    # -----------------------
    def extraer(self, texto):

        # 🔥 NORMALIZACIÓN
        texto = self.normalizar(texto)
        texto = self.aplicar_diccionario(texto)

        entidades = {
            "sintomas": [],
            "medicamentos": [],
            "alergias": [],
            "diagnosticos": []
        }

        # -----------------------
        # 🧠 SÍNTOMAS (REGLAS)
        # -----------------------
        sintomas_base = [
            "dolor", "fiebre", "tos", "mareo",
            "dificultad respiratoria", "náuseas"
        ]

        for s in sintomas_base:
            if s in texto:
                entidades["sintomas"].append(s)

        # 🔥 frase clínica fuerte
        if "dolor torácico opresivo" in texto:
            entidades["sintomas"].append("dolor torácico opresivo")

        # -----------------------
        # 📚 SÍNTOMAS (DICCIONARIO)
        # -----------------------
        for sintoma in DICCIONARIO.get("sintomas", []):
            if sintoma in texto:
                entidades["sintomas"].append(sintoma)

        # -----------------------
        # 💊 MEDICAMENTOS (REGEX)
        # -----------------------

        meds1 = re.findall(r"\b([a-záéíóúñ]+)\s*(\d+\s?mg)\b", texto)
        meds2 = re.findall(r"\b(\d+\s?mg)\s*(?:de)?\s*([a-záéíóúñ]+)\b", texto)

        medicamentos_struct = []

        for nombre, dosis in meds1:
            medicamentos_struct.append({
                "nombre": nombre,
                "dosis": dosis
            })

        for dosis, nombre in meds2:
            medicamentos_struct.append({
                "nombre": nombre,
                "dosis": dosis
            })

        # -----------------------
        # 📚 MEDICAMENTOS (DICCIONARIO)
        # -----------------------
        for med in DICCIONARIO.get("medicamentos", []):
            if med in texto:
                if not any(m["nombre"] == med for m in medicamentos_struct):
                    medicamentos_struct.append({
                        "nombre": med,
                        "dosis": "no especificada"
                    })

        entidades["medicamentos"] = medicamentos_struct

        # -----------------------
        # ⚠️ ALERGIAS
        # -----------------------

        match_alergia = re.findall(
            r"al[eé]rgic[oa]\s+a\s+(?:la\s+|el\s+)?([a-záéíóúñ]+)",
            texto
        )

        entidades["alergias"].extend(match_alergia)

        # 📚 diccionario alergias
        for alergia in DICCIONARIO.get("alergias", []):
            if alergia in texto:
                entidades["alergias"].append(alergia)

        # -----------------------
        # 🧾 DIAGNÓSTICOS
        # -----------------------

        match_dx = re.findall(
            r"(?:diagn[oó]stico[:]?|dx[:]?)(\s*[a-záéíóúñ\s]+?)(?:\.|,|$)",
            texto
        )

        # 📚 diccionario diagnósticos
        for dx in DICCIONARIO.get("diagnosticos", []):
            if dx in texto:
                match_dx.append(dx)

        entidades["diagnosticos"] = [d.strip() for d in match_dx if d.strip()]

        # -----------------------
        # 🤖 NER (COMPLEMENTO)
        # -----------------------

        try:
            resultados = ner_pipeline(texto)

            for r in resultados:
                palabra = r["word"].lower()

                if "dolor" in palabra:
                    entidades["sintomas"].append(palabra)

        except:
            pass  # evita romper la API

        # -----------------------
        # 🧹 LIMPIEZA FINAL
        # -----------------------

        for k in ["sintomas", "alergias", "diagnosticos"]:
            entidades[k] = list(set(entidades[k]))

        return entidades