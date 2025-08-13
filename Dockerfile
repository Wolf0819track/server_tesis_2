# Usar una versión de Python compatible con las dependencias (ej. TensorFlow 2.15)
FROM python:3.9-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar primero el archivo de requerimientos para aprovechar el caché de capas de Docker
COPY requirements.txt .

# Instalar las dependencias, incluyendo el modelo de spacy desde la URL
# --no-cache-dir ayuda a reducir el tamaño final de la imagen
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos de la aplicación al contenedor
COPY . .

# El puerto de la aplicación que Hugging Face expondrá.
# Usamos 7860 como un estándar común en la plataforma.
EXPOSE 7860

# Comando para ejecutar la aplicación usando un servidor de producción (gunicorn).
# Se enlaza a la variable $PORT que Hugging Face provee automáticamente.
# --workers 1 es ideal para planes gratuitos con CPU limitado.
# --timeout 120 da más tiempo a las peticiones para procesar los modelos.
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--timeout", "120", "server:app"]
