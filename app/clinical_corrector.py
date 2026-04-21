# app/clinical_corrector.py

import re

class CorrectorClinico:

    # -----------------------
    # 🧼 LIMPIEZA (SIN SEMÁNTICA)
    # -----------------------
    def limpiar(self, texto):

        texto = texto.lower()

        ruido = [
            "eh", "mmm", "este", "pues",
            "fíle", "creado", "huevo", "bolinos"
        ]

        for r in ruido:
            texto = texto.replace(r, " ")

        texto = re.sub(r"[^\w\sáéíóúñ]", " ", texto)
        texto = re.sub(r"\s+", " ", texto)

        return texto.strip()

    # -----------------------
    # 🧠 NORMALIZACIÓN (SOLO EQUIVALENCIAS)
    # -----------------------
    def normalizar(self, texto):

        reglas = {
            "torasico": "torácico",
            "toracico": "torácico",
            "disneaa": "disnea",
            "aspedina": "aspirina",
            "mililitros": "mg",
            "mililitro": "mg",
            "pacente": "paciente"
        }

        for k, v in reglas.items():
            texto = texto.replace(k, v)

        return texto

    # -----------------------
    # 🚀 PIPELINE FINAL
    # -----------------------
    def corregir(self, texto):

        texto = self.limpiar(texto)
        texto = self.normalizar(texto)

        return texto