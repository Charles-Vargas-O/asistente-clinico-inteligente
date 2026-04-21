import whisper

class SpeechToText:
    def __init__(self):
        self.model = whisper.load_model("base")

    def transcribir(self, audio_path):
        result = self.model.transcribe(audio_path, language="es")
        return result["text"].lower()