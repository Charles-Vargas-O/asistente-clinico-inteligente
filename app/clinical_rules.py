class MotorClinico:

    def evaluar(self, datos):

        alertas = []
        sugerencias = []

        sintomas = datos.get("sintomas", [])
        alergias = datos.get("alergias", [])
        diagnosticos = datos.get("diagnosticos", [])
        medicamentos = datos.get("medicamentos", [])

        nombres_meds = [m["nombre"] for m in medicamentos]

        # 1. ALERTAS DE ALERGIA
        for alergia in alergias:
            for med in nombres_meds:
                if alergia in med:
                    alertas.append(f"⚠️ Posible reacción alérgica a {med}")

        # 2. VALIDACIÓN BÁSICA
        if "dolor" in sintomas and "gastritis" in diagnosticos:
            sugerencias.append("Diagnóstico consistente con síntomas gastrointestinales")

        # 3. REGLAS CLÍNICAS SIMPLES
        if "náuseas" in sintomas and "dolor" in sintomas:
            sugerencias.append("Evaluar posible irritación gastrointestinal")

        # 4. FALTA DE INFORMACIÓN
        if not diagnosticos:
            alertas.append("⚠️ No se detectó diagnóstico")

        if not medicamentos:
            alertas.append("⚠️ No se detectaron medicamentos")

        return {
            "alertas": alertas,
            "sugerencias": sugerencias
        }