import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class ModeloDiagnostico:

    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.entrenado = False

    def entrenar(self):

        with open("data/casos.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        textos = [" ".join(d["sintomas"]) for d in data]
        etiquetas = [d["diagnostico"] for d in data]

        X = self.vectorizer.fit_transform(textos)
        self.model.fit(X, etiquetas)

        self.entrenado = True

    def predecir(self, sintomas):

        if not self.entrenado:
            return None

        texto = " ".join(sintomas)
        X = self.vectorizer.transform([texto])

        return self.model.predict(X)[0]