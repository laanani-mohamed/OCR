# Dockerfile pour le déploiement d'un projet OCR avec Python 3.10
FROM python:3.10-slim

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Créer et définir le répertoire de travail
WORKDIR /app

# Mettre à jour pip à la dernière version
RUN pip install --upgrade pip

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Créer des répertoires pour stocker les modèles
RUN mkdir -p /root/.paddleocr/ /root/.cache/huggingface/ /data/input /data/output

# Fonction pour vérifier si le modèle est déjà téléchargé
RUN echo "Vérification et téléchargement des modèles..." && \
    python -c "from paddleocr import PaddleOCR; \
               try: \
                   ocr = PaddleOCR()  # Teste si le modèle est déjà téléchargé \
               except Exception as e: \
                   print('Erreur PaddleOCR:', str(e))" || \
    echo "PaddleOCR modèle déjà téléchargé ou erreur rencontrée, réessayer..." && \
    python -c "from paddleocr import PaddleOCR; PaddleOCR()"

# Réessayer le téléchargement de TrOCR
RUN echo "Vérification et téléchargement des modèles TrOCR..." && \
    python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel; \
               try: \
                   TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten'); \
                   VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten'); \
               except Exception as e: \
                   print('Erreur TrOCR:', str(e))" || \
    echo "TrOCR modèle déjà téléchargé ou erreur rencontrée, réessayer..." && \
    python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel; \
               TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten'); \
               VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')"

# Exposer le port pour l'API
EXPOSE 9876

# Commande par défaut pour démarrer l'application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9876"]
