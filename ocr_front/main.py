import streamlit as st
import requests
import json
import pandas as pd
import os
import tempfile

# Configuration de la page
st.set_page_config(
    page_title="Application OCR de Documents",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Titre de l'application
st.title("Application OCR de Documents")
st.write("Téléchargez un PDF ou une image pour extraction automatique des données")

# URL de l'API - récupérer depuis la variable d'environnement ou utiliser une valeur par défaut
API_URL = os.environ.get("API_URL", "http://ocr-api:9876")

# Fonction pour envoyer le fichier à l'API
def process_file(file, file_type):
    # Créer un fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Déterminer l'endpoint en fonction du type de fichier
    if file_type == "pdf":
        endpoint = f"{API_URL}/process_pdf/"
    else:
        endpoint = f"{API_URL}/process_image/"
    
    # Préparer les données pour l'API
    files = {'file': (file.name, open(tmp_file_path, 'rb'), f'application/{file_type}')}
    
    try:
        # Envoyer la requête à l'API
        response = requests.post(endpoint, files=files)
        
        # Supprimer le fichier temporaire
        os.unlink(tmp_file_path)
        
        # Vérifier la réponse
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de la communication avec l'API: {str(e)}")
        # Supprimer le fichier temporaire en cas d'erreur
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        return None

# Section d'upload de fichier
st.subheader("Téléchargement de fichier")

# Sélection du type de fichier
file_type = st.radio(
    "Sélectionnez le type de document",
    options=["PDF", "Image"],
    horizontal=True
)

# Upload du fichier selon le type sélectionné
if file_type == "PDF":
    uploaded_file = st.file_uploader("Téléchargez un fichier PDF", type=["pdf"])
else:
    uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

# Bouton pour traiter le fichier
if uploaded_file is not None:
    st.write(f"Fichier sélectionné: {uploaded_file.name}")
    
    if st.button("Analyser le document"):
        with st.spinner("Traitement en cours..."):
            # Traiter le fichier et obtenir les résultats
            results = process_file(uploaded_file, "pdf" if file_type == "PDF" else "image")
            
            if results:
                st.success("Traitement terminé avec succès!")
                
                # Afficher les résultats en fonction du type de fichier
                if file_type == "PDF":
                    # Afficher les résultats du PDF
                    if "pdf_results" in results:
                        pdf_results = results["pdf_results"]
                        
                        # Créer des onglets pour les différentes sections
                        tab1, tab2, tab3, tab4 = st.tabs(["Véhicule A", "Véhicule B", "En-tête", "Tous les véhicules"])
                        
                        # Onglet Véhicule A
                        with tab1:
                            st.header("Informations du Véhicule A")
                            if "vehicule_A" in pdf_results:
                                display_vehicle_data(pdf_results["vehicule_A"])
                            else:
                                st.warning("Aucune information trouvée pour le véhicule A")
                        
                        # Onglet Véhicule B
                        with tab2:
                            st.header("Informations du Véhicule B")
                            if "vehicule_B" in pdf_results:
                                display_vehicle_data(pdf_results["vehicule_B"])
                            else:
                                st.warning("Aucune information trouvée pour le véhicule B")
                        
                        # Onglet En-tête
                        with tab3:
                            st.header("En-tête du document")
                            if "en_tete" in pdf_results:
                                display_section_data(pdf_results["en_tete"])
                            else:
                                st.warning("Aucune information trouvée pour l'en-tête")
                        
                        # Onglet Tous les véhicules
                        with tab4:
                            st.header("Comparaison des véhicules")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Véhicule A")
                                if "vehicule_A" in pdf_results:
                                    display_vehicle_data(pdf_results["vehicule_A"])
                                else:
                                    st.warning("Aucune information trouvée pour le véhicule A")
                            
                            with col2:
                                st.subheader("Véhicule B")
                                if "vehicule_B" in pdf_results:
                                    display_vehicle_data(pdf_results["vehicule_B"])
                                else:
                                    st.warning("Aucune information trouvée pour le véhicule B")
                    else:
                        st.error("Format de réponse PDF invalide")
                
                else:  # Résultats d'image
                    # Afficher les résultats de l'image
                    if "extracted_info" in results:
                        extracted_info = results["extracted_info"]
                        
                        st.header("Informations Extraites")
                        
                        # Créer un DataFrame pour afficher les informations
                        data = {
                            "Information": ["Numéro de police", "Date de début", "Date de fin"],
                            "Valeur": [
                                extracted_info.get("numero_police", "Non détecté"),
                                extracted_info.get("date_debut", "Non détectée"),
                                extracted_info.get("date_fin", "Non détectée")
                            ]
                        }
                        
                        df = pd.DataFrame(data)
                        st.table(df)
                    else:
                        st.error("Format de réponse image invalide")

# Fonction pour afficher les données d'un véhicule
def display_vehicle_data(vehicle_data):
    if not vehicle_data:
        st.warning("Aucune donnée disponible")
        return
    
    for section, data in vehicle_data.items():
        st.subheader(f"Partie {section.split('_')[1] if '_' in section else section}")
        
        if isinstance(data, list):
            # Afficher les données ligne par ligne
            for i, line in enumerate(data):
                st.text(f"{i+1}. {line}")
        else:
            # Afficher les données sous forme de JSON
            st.json(data)
        
        st.markdown("---")

# Fonction pour afficher les données d'une section
def display_section_data(section_data):
    if not section_data:
        st.warning("Aucune donnée disponible")
        return
    
    for section, data in section_data.items():
        st.subheader(f"{section}")
        
        if isinstance(data, list):
            # Afficher les données ligne par ligne
            for i, line in enumerate(data):
                st.text(f"{i+1}. {line}")
        else:
            # Afficher les données sous forme de JSON
            st.json(data)
        
        st.markdown("---")

# Ajouter des informations supplémentaires
st.sidebar.title("À propos")
st.sidebar.info(
    """
    Cette application utilise l'OCR pour extraire automatiquement 
    les informations des documents PDF et images.
    
    Technologies utilisées:
    - Streamlit pour l'interface
    - TrOCR pour les PDFs
    - PaddleOCR pour les images
    """
)

# Afficher la configuration de connexion dans la barre latérale pour le débogage
st.sidebar.title("Configuration")
st.sidebar.text(f"API URL: {API_URL}")

# Footer
st.markdown("---")
st.markdown("© 2025 Application OCR de Documents")