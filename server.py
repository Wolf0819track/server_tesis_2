import os
import cv2
import tempfile
import shutil
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from spellchecker import SpellChecker
from itertools import groupby
from gtts import gTTS
import base64
import spacy

# =========================
# Configuración inicial
# =========================
app = Flask(__name__)

# Carpeta temporal en la nube
UPLOAD_FOLDER = tempfile.mkdtemp()
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print("📁 Carpeta temporal:", UPLOAD_FOLDER)

# Cargar modelo y escalador una sola vez
MODEL_PATH = "modelo_X_5.h5"
SCALER_PATH = "escalador_5.pkl"

print("🔄 Cargando modelo y escalador...")
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Modelo y escalador cargados.")

# Cargar procesadores de texto
nlp = spacy.load("es_core_news_sm")
spell = SpellChecker(language='es')

# =========================
# Endpoint raíz (Render / Railway health check)
# =========================
@app.route("/")
def home():
    return "Servidor en línea 🚀"

# =========================
# Función para limpiar directorio
# =========================
def limpiar_directorio(path):
    for archivo in os.listdir(path):
        archivo_path = os.path.join(path, archivo)
        if os.path.isfile(archivo_path):
            os.remove(archivo_path)

# =========================
# Endpoint para recibir imágenes
# =========================
@app.route("/recibir_imagenes", methods=["POST"])
def recibir_imagenes():
    files = request.files.getlist("imagenes")
    if not files:
        return jsonify({"error": "No se recibieron archivos"}), 400

    limpiar_directorio(UPLOAD_FOLDER)

    for idx, file in enumerate(files):
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, f"{idx}_{filename}")
            file.save(file_path)

    palabra_detectada, path_audio = analizar_frames_en_directorio(UPLOAD_FOLDER)

    return jsonify({
        "mensaje": "Procesado correctamente",
        "palabra": palabra_detectada,
        "audio": {
            "audio": base64.b64encode(open(path_audio, "rb").read()).decode("utf-8")
        }
    })

# =========================
# Procesamiento de frames
# =========================
def analizar_frames_en_directorio(directorio):
    imagenes = sorted([os.path.join(directorio, img) for img in os.listdir(directorio) if img.endswith(".jpg")])
    if not imagenes:
        return "", ""

    predicciones = []
    for img_path in imagenes:
        frame = cv2.imread(img_path)
        pred = procesar_frame(frame)
        if pred:
            predicciones.append(pred)

    palabra = "".join(k for k, _ in groupby(predicciones))
    palabra_limpia = palabra.upper()

    # Corrección ortográfica solo si no es nombre propio
    doc = nlp(palabra_limpia)
    for token in doc:
        if token.ent_type_ != "PER":
            palabra_limpia = spell.correction(palabra_limpia) or palabra_limpia

    # Generar audio
    path_audio = os.path.join(UPLOAD_FOLDER, "voz.mp3")
    tts = gTTS(text=palabra_limpia, lang="es")
    tts.save(path_audio)

    return palabra_limpia, path_audio

# =========================
# Procesar un solo frame
# =========================
def procesar_frame(frame):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None

        puntos = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                puntos.extend([lm.x, lm.y, lm.z])

        if len(puntos) != 126:
            return None

        X = scaler.transform([puntos])
        prediccion = model.predict(X, verbose=0)
        return chr(np.argmax(prediccion) + ord("A"))

# =========================
# Iniciar servidor
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

