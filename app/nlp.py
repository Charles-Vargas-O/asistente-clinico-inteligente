import spacy

class MotorNLP:
    def __init__(self):
        try:
            self.nlp = spacy.load("modelo")
        except:
            self.nlp = spacy.load("es_core_news_md")

    def extraer(self, texto):
        doc = self.nlp(texto)

        entidades = {
            "alergias": [],
            "medicamentos": [],
            "sintomas": [],
            "diagnosticos": []
        }

        for ent in doc.ents:
            if ent.label_.lower() == "alergia":
                entidades["alergias"].append(ent.text)
            elif ent.label_.lower() == "medicamento":
                entidades["medicamentos"].append(ent.text)
            elif ent.label_.lower() == "sintoma":
                entidades["sintomas"].append(ent.text)
            elif ent.label_.lower() == "diagnostico":
                entidades["diagnosticos"].append(ent.text)

        return entidades