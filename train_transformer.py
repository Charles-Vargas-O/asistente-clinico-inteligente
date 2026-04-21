import spacy
from spacy.training import Example

# 🔥 Modelo transformer
nlp = spacy.load("es_core_news_trf")

# Agregar NER si no existe
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Etiquetas clínicas
labels = ["SINTOMA", "MEDICAMENTO", "ALERGIA", "DIAGNOSTICO"]
for label in labels:
    ner.add_label(label)

# Dataset MÁS REALISTA
TRAIN_DATA = [
    ("Paciente masculino 45 años con dolor abdominal intenso y náuseas. Toma omeprazol 20 mg. Alérgico a penicilina. Diagnóstico gastritis",
     {"entities": [
         (33, 57, "SINTOMA"),
         (60, 67, "SINTOMA"),
         (75, 92, "MEDICAMENTO"),
         (106, 117, "ALERGIA"),
         (131, 140, "DIAGNOSTICO")
     ]}),
]

# Entrenamiento
optimizer = nlp.resume_training()

for i in range(20):
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], sgd=optimizer)

print("Modelo transformer entrenado")

# Guardar
nlp.to_disk("modelo_trf")