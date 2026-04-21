import spacy
from spacy.training import Example

# Crear modelo en blanco en español
nlp = spacy.blank("es")

# Crear componente NER
ner = nlp.add_pipe("ner")

# Etiquetas clínicas
ner.add_label("SINTOMA")
ner.add_label("MEDICAMENTO")
ner.add_label("ALERGIA")
ner.add_label("DIAGNOSTICO")

# Dataset de entrenamiento (puedes ampliarlo después)
TRAIN_DATA = [
    # SÍNTOMAS
    ("Paciente con dolor abdominal intenso", {"entities": [(13, 36, "SINTOMA")]}),
    ("Presenta náuseas y vómito", {"entities": [(9, 16, "SINTOMA"), (19, 25, "SINTOMA")]}),
    ("Refiere fiebre alta", {"entities": [(8, 18, "SINTOMA")]}),

    # MEDICAMENTOS
    ("Toma omeprazol 20 mg diario", {"entities": [(5, 22, "MEDICAMENTO")]}),
    ("Consume losartan 50 mg", {"entities": [(8, 24, "MEDICAMENTO")]}),
    ("Tratamiento con paracetamol 500 mg", {"entities": [(17, 37, "MEDICAMENTO")]}),

    # ALERGIAS
    ("Alérgico a penicilina", {"entities": [(12, 23, "ALERGIA")]}),
    ("Alergia al ibuprofeno", {"entities": [(11, 22, "ALERGIA")]}),
    ("Paciente con alergia a amoxicilina", {"entities": [(24, 36, "ALERGIA")]}),

    # DIAGNÓSTICOS
    ("Diagnóstico gastritis", {"entities": [(12, 21, "DIAGNOSTICO")]}),
    ("Dx diabetes mellitus tipo 2", {"entities": [(3, 30, "DIAGNOSTICO")]}),
    ("Impresión diagnóstica hipertensión arterial", {"entities": [(22, 46, "DIAGNOSTICO")]}),

    # CASOS COMPLETOS
    ("Paciente con dolor abdominal. Toma omeprazol 20 mg. Alérgico a penicilina. Diagnóstico gastritis", {"entities": [
     (17, 32, "SINTOMA"),
     (39, 56, "MEDICAMENTO"),
     (69, 80, "ALERGIA"),
     (94, 103, "DIAGNOSTICO") ]}),

    ("Refiere náuseas. Consume losartan 50 mg. Alergia a ibuprofeno", {"entities": [
     (8, 15, "SINTOMA"),
     (26, 42, "MEDICAMENTO"),
     (53, 64, "ALERGIA")]})
     
]

# Entrenamiento
optimizer = nlp.begin_training()

for i in range(80):
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], sgd=optimizer)

print("Modelo entrenado")

# Guardar modelo
nlp.to_disk("modelo")