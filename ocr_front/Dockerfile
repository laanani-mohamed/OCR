# Utiliser une image Python officielle allégée
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY main.py /app/
COPY requirements.txt /app/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port Streamlit
EXPOSE 9898

# Définir les variables d'environnement pour Streamlit
ENV STREAMLIT_SERVER_PORT=9898
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Démarrer l'application avec le bon fichier
CMD ["streamlit", "run", "main.py", "--server.port=9898", "--server.address=0.0.0.0"]
