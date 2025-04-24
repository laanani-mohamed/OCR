FROM python:3.10-slim

# Créer le dossier de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copier requirements.txt et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- PRÉCHARGEMENT DES MODÈLES ----------
# PaddleOCR
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='fr')"

# TrOCR
RUN python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel; \
               TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten'); \
               VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')"

# Doctr
RUN python -c "from doctr.models import fast_tiny; fast_tiny(pretrained=True)"
# ------------------------------------------------

# Copier les fichiers sources
COPY app.py .
COPY processor_image.py .
COPY processor_pdf.py .

# Exposer le port Flask
EXPOSE 5000

# Variables d'environnement
ENV PORT=5000
ENV FLASK_DEBUG=False

# Lancer l'application avec Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--timeout", "300", "--workers", "1"]