from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import csv
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import json
from datetime import datetime
import numpy as np
import fitz  # PyMuPDF

class ConstatProcessor:
    # Modèles chargés une seule fois au niveau de la classe
    _processor = None
    _model = None
    
    def __init__(self):
        # Vérifier si CUDA est disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Charger les modèles une seule fois
        if ConstatProcessor._processor is None:
            ConstatProcessor._processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
            ConstatProcessor._model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to(self.device)
        
        self.processor = ConstatProcessor._processor
        self.model = ConstatProcessor._model
        
        # Définir toutes les parties à analyser
        self.parties = [
            # En-tête
            {
                "name": "Date_Accident",
                "x": 200, "y": 160, "w": 400, "h": 80,
                "line_height": 80,
                "mapping": [
                    {"structure_path": ["en_tete", "date_accident"], "ligne_num": 1, "description": "Date de l'accident"}
                ]
            },
            {
                "name": "Heure_Accident",
                "x": 590, "y": 160, "w": 330, "h": 80,
                "line_height": 80,
                "mapping": [
                    {"structure_path": ["en_tete", "heure_accident"], "ligne_num": 1, "description": "Heure de l'accident"}
                ]
            },
            {
                "name": "Lieu_Accident",
                "x": 930, "y": 160, "w": 400, "h": 80,
                "line_height": 80,
                "mapping": [
                    {"structure_path": ["en_tete", "lieu_accident"], "ligne_num": 1, "description": "Lieu de l'accident"}
                ]
            },
            {
                "name": "Lieu_Precis",
                "x": 140, "y": 220, "w": 1200, "h": 60,
                "line_height": 60,
                "mapping": [
                    {"structure_path": ["en_tete", "lieu_accident_precis"], "ligne_num": 1, "description": "Lieu précis de l'accident"}
                ]
            },

            # Partie A
            {
                "name": "Telephone_A",
                "x": 0, "y": 330, "w": 480, "h": 110,
                "line_height": 110,
                "mapping": [
                    {"structure_path": ["partie_A", "vehicule", "telephone"], "ligne_num": 1, "description": "Téléphone"}
                ]
            },
            {
                "name": "Vehicule_A",
                "x": 0, "y": 460, "w": 480, "h": 230,
                "line_height": 45,
                "mapping": [
                    {"structure_path": ["partie_A", "vehicule", "type_vehicule"], "ligne_num": 1, "description": "Type de véhicule"},
                    {"structure_path": ["partie_A", "vehicule", "n_immatriculation"], "ligne_num": 2, "description": "N° d'immatriculation"},
                    {"structure_path": ["partie_A", "vehicule", "venant_de"], "ligne_num": 3, "description": "Venant de"},
                    {"structure_path": ["partie_A", "vehicule", "allant_a"], "ligne_num": 4, "description": "Allant à"}
                ]
            },
            {
                "name": "Assure_A",
                "x": 0, "y": 690, "w": 480, "h": 335,
                "line_height": 50,
                "mapping": [
                    {"structure_path": ["partie_A", "assurance", "nom"], "ligne_num": 1, "description": "Nom"},
                    {"structure_path": ["partie_A", "assurance", "prenom"], "ligne_num": 2, "description": "Prénom"},
                    {"structure_path": ["partie_A", "assurance", "adresse"], "ligne_num": 3, "description": "Adresse"},
                    {"structure_path": ["partie_A", "assurance", "n_attestation"], "ligne_num": 4, "description": "N° d'attestation"},
                    {"structure_path": ["partie_A", "assurance", "n_police"], "ligne_num": 5, "description": "N° de police"},
                    {"structure_path": ["partie_A", "assurance", "date_validite"], "ligne_num": 6, "description": "Date de validité"}
                ]
            },
            {
                "name": "Conducteur_A",
                "x": 0, "y": 1050, "w": 480, "h": 300,
                "line_height": 45,
                "mapping": [
                    {"structure_path": ["partie_A", "conducteur", "nom"], "ligne_num": 1, "description": "Nom"},
                    {"structure_path": ["partie_A", "conducteur", "prenom"], "ligne_num": 2, "description": "Prénom"},
                    {"structure_path": ["partie_A", "conducteur", "adresse"], "ligne_num": 3, "description": "Adresse"},
                    {"structure_path": ["partie_A", "conducteur", "n_permis"], "ligne_num": 4, "description": "N° du permis"},
                    {"structure_path": ["partie_A", "conducteur", "date_validite_permis"], "ligne_num": 5, "description": "Date de validité du permis"}
                ]
            },
            {
                "name": "Degat_Observation_A",
                "x": 0, "y": 1550, "w": 480, "h": 300,
                "line_height": 300,
                "mapping": [
                    {"structure_path": ["partie_A", "degat_apparence", "description"], "ligne_num": 1, "description": "Description des dégâts apparents"}
                ]
            },

            # Partie B
            {
                "name": "Telephone_B",
                "x": 1020, "y": 330, "w": 480, "h": 110,
                "line_height": 110,
                "mapping": [
                    {"structure_path": ["partie_B", "vehicule", "telephone"], "ligne_num": 1, "description": "Téléphone"}
                ]
            },
            {
                "name": "Vehicule_B",
                "x": 1020, "y": 460, "w": 480, "h": 230,
                "line_height": 45,
                "mapping": [
                    {"structure_path": ["partie_B", "vehicule", "type_vehicule"], "ligne_num": 1, "description": "Type de véhicule"},
                    {"structure_path": ["partie_B", "vehicule", "n_immatriculation"], "ligne_num": 2, "description": "N° d'immatriculation"},
                    {"structure_path": ["partie_B", "vehicule", "venant_de"], "ligne_num": 3, "description": "Venant de"},
                    {"structure_path": ["partie_B", "vehicule", "allant_a"], "ligne_num": 4, "description": "Allant à"}
                ]
            },
            {
                "name": "Assure_B",
                "x": 1020, "y": 690, "w": 480, "h": 335,
                "line_height": 50,
                "mapping": [
                    {"structure_path": ["partie_B", "assurance", "nom"], "ligne_num": 1, "description": "Nom"},
                    {"structure_path": ["partie_B", "assurance", "prenom"], "ligne_num": 2, "description": "Prénom"},
                    {"structure_path": ["partie_B", "assurance", "adresse"], "ligne_num": 3, "description": "Adresse"},
                    {"structure_path": ["partie_B", "assurance", "n_attestation"], "ligne_num": 4, "description": "N° d'attestation"},
                    {"structure_path": ["partie_B", "assurance", "n_police"], "ligne_num": 5, "description": "N° de police"},
                    {"structure_path": ["partie_B", "assurance", "date_validite"], "ligne_num": 6, "description": "Date de validité"}
                ]
            },
            {
                "name": "Conducteur_B",
                "x": 1020, "y": 1000, "w": 480, "h": 340,
                "line_height": 50,
                "mapping": [
                    {"structure_path": ["partie_B", "conducteur", "nom"], "ligne_num": 1, "description": "Nom"},
                    {"structure_path": ["partie_B", "conducteur", "prenom"], "ligne_num": 2, "description": "Prénom"},
                    {"structure_path": ["partie_B", "conducteur", "adresse"], "ligne_num": 3, "description": "Adresse"},
                    {"structure_path": ["partie_B", "conducteur", "n_permis"], "ligne_num": 4, "description": "N° du permis"},
                    {"structure_path": ["partie_B", "conducteur", "date_validite_permis"], "ligne_num": 5, "description": "Date de validité du permis"}
                ]
            },
            {
                "name": "Degat_Observation_B",
                "x": 1020, "y": 1550, "w": 480, "h": 300,
                "line_height": 300,
                "mapping": [
                    {"structure_path": ["partie_B", "degat_apparence", "description"], "ligne_num": 1, "description": "Description des dégâts apparents"}
                ]
            }
        ]

    def init_structure_constat(self):
        """Structure de données pour le constat amiable"""
        return {
            "en_tete": {
                "date_accident": None,
                "heure_accident": None,
                "lieu_accident": None,
                "lieu_accident_precis": None
            },
            "partie_A": {
                "vehicule": {
                    "telephone": None,
                    "type_vehicule": None,
                    "n_immatriculation": None,
                    "venant_de": None,
                    "allant_a": None
                },
                "assurance": {
                    "nom": None,
                    "prenom": None,
                    "adresse": None,
                    "n_attestation": None,
                    "n_police": None,
                    "date_validite": None
                },
                "conducteur": {
                    "nom": None,
                    "prenom": None,
                    "adresse": None,
                    "n_permis": None,
                    "date_validite_permis": None
                },
                "degat_apparence": {
                    "description": None
                }
            },
            "partie_B": {
                "vehicule": {
                    "telephone": None,
                    "type_vehicule": None,
                    "n_immatriculation": None,
                    "venant_de": None,
                    "allant_a": None
                },
                "assurance": {
                    "nom": None,
                    "prenom": None,
                    "adresse": None,
                    "n_attestation": None,
                    "n_police": None,
                    "date_validite": None
                },
                "conducteur": {
                    "nom": None,
                    "prenom": None,
                    "adresse": None,
                    "n_permis": None,
                    "date_validite_permis": None
                },
                "degat_apparence": {
                    "description": None
                }
            }
        }

    def set_nested_dict_value(self, d, path, value):
        """Met à jour un dictionnaire imbriqué en suivant un chemin spécifié"""
        current = d
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def visualiser_roi(self, image, partie):
        """Visualise une ROI en dessinant un rectangle sur l'image"""
        draw = ImageDraw.Draw(image)
        x, y, w, h = partie["x"], partie["y"], partie["w"], partie["h"]
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)
        return image

    def extraire_et_analyser_partie(self, image, partie, visualiser=False):
        """Extrait une partie de l'image et fait la reconnaissance de texte par ligne"""
        try:
            # Extraire les paramètres
            x, y, w, h = partie["x"], partie["y"], partie["w"], partie["h"]
            line_height = partie["line_height"]
            name = partie["name"]

            # Vérifier si les coordonnées sont dans les limites
            img_width, img_height = image.size
            if x + w > img_width or y + h > img_height:
                # Ajuster les coordonnées
                x = min(x, img_width - 1)
                y = min(y, img_height - 1)
                w = min(w, img_width - x)
                h = min(h, img_height - y)

            # Extraire la partie
            partie_image = image.crop((x, y, x + w, y + h))

            # S'assurer que l'image est en mode RGB
            if partie_image.mode != 'RGB':
                partie_image = partie_image.convert('RGB')

            # Découper la partie en lignes selon line_height
            num_lines = max(1, int(h / line_height))
            lignes_resultats = []

            # Parcourir chaque ligne
            for i in range(num_lines):
                # Calculer les coordonnées de la ligne
                ligne_y = i * line_height
                ligne_h = min(line_height, h - ligne_y)

                if ligne_h <= 0:
                    break

                # Extraire la ligne
                ligne_image = partie_image.crop((0, ligne_y, w, ligne_y + ligne_h))

                # S'assurer que la ligne est en mode RGB
                if ligne_image.mode != 'RGB':
                    ligne_image = ligne_image.convert('RGB')

                # Reconnaissance de texte avec TrOCR
                pixel_values = self.processor(images=ligne_image, return_tensors="pt").pixel_values.to(self.device)
                generated_ids = self.model.generate(pixel_values)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Nettoyer un peu le texte
                generated_text = generated_text.strip()

                # Ajouter le résultat
                ligne_info = {
                    "partie": name,
                    "ligne_num": i+1,
                    "texte": generated_text
                }
                lignes_resultats.append(ligne_info)

            return {
                "partie_name": name,
                "partie_image": partie_image,
                "lignes": lignes_resultats
            }

        except Exception as e:
            return None

    def traiter_image_complete(self, image_path=None, image_redimensionnee=None, visualiser_roi_flag=False):
        """Traite une image complète avec toutes les ROI définies"""
        try:
            # Charger l'image - soit à partir du chemin, soit utiliser celle fournie
            if image_path and not image_redimensionnee:
                image = Image.open(image_path)
            elif image_redimensionnee is not None:
                image = image_redimensionnee
            else:
                raise ValueError("Vous devez fournir soit un chemin d'image, soit une image redimensionnée")

            # Créer une copie pour la visualisation si nécessaire
            image_avec_roi = image.copy() if visualiser_roi_flag else None

            # Initialiser la structure de données pour les résultats
            structure_constat = self.init_structure_constat()
            resultats_parties = []

            # Créer des dossiers pour les sorties
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"resultats_ocr_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Traiter chaque partie
            for partie in self.parties:
                # Dessiner le rectangle ROI sur l'image si demandé
                if visualiser_roi_flag:
                    image_avec_roi = self.visualiser_roi(image_avec_roi, partie)

                # Analyser la partie
                resultat = self.extraire_et_analyser_partie(image, partie, visualiser=False)

                if resultat:
                    resultats_parties.append(resultat)

                    # Mettre à jour la structure constat
                    for ligne in resultat["lignes"]:
                        ligne_num = ligne["ligne_num"]
                        texte = ligne["texte"]

                        # Trouver le mapping correspondant
                        for mapping_item in partie.get("mapping", []):
                            if mapping_item["ligne_num"] == ligne_num:
                                structure_path = mapping_item["structure_path"]
                                self.set_nested_dict_value(structure_constat, structure_path, texte)

            # Sauvegarder l'image avec les ROI visualisées si demandé
            roi_overlay_path = None
            if visualiser_roi_flag:
                roi_overlay_path = os.path.join(output_dir, "image_avec_roi.png")
                image_avec_roi.save(roi_overlay_path)

            # Exporter les résultats en JSON
            json_path = os.path.join(output_dir, "resultats_structure.json")
            with open(json_path, "w", encoding='utf-8') as f:
                json.dump(structure_constat, f, ensure_ascii=False, indent=4)

            # Exporter les résultats en CSV (comme le format de sortie secondaire)
            csv_path = os.path.join(output_dir, "resultats_ocr.csv")
            with open(csv_path, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Partie", "Ligne", "Description", "Texte", "Chemin Structure"])

                for partie_resultat in resultats_parties:
                    partie_name = partie_resultat["partie_name"]

                    # Trouver la partie correspondante dans la définition
                    partie_def = next((p for p in self.parties if p["name"] == partie_name), None)

                    for ligne in partie_resultat["lignes"]:
                        ligne_num = ligne["ligne_num"]
                        texte = ligne["texte"]

                        # Trouver la description et le chemin de structure
                        description = "N/A"
                        chemin_structure = "N/A"
                        if partie_def:
                            for mapping_item in partie_def.get("mapping", []):
                                if mapping_item["ligne_num"] == ligne_num:
                                    description = mapping_item["description"]
                                    chemin_structure = ".".join(mapping_item["structure_path"])

                        writer.writerow([partie_name, ligne_num, description, texte, chemin_structure])

            return {
                "structure_constat": structure_constat,
                "json_path": json_path
            }

        except Exception as e:
            return {"error": str(e)}

    def extraire_zone_normalisee(self, image_array, coords, margin=0):
        """Extrait une zone normalisée d'une image"""
        # Vérifier si l'image est déjà normalisée (0-255) ou non (0-1)
        if np.issubdtype(image_array.dtype, np.floating) and image_array.max() <= 1.0:
            # Si l'image a des valeurs entre 0 et 1, la normaliser correctement
            image_array = (image_array * 255).astype(np.uint8)
        elif not np.issubdtype(image_array.dtype, np.uint8):
            # Si ce n'est pas déjà uint8, convertir
            image_array = image_array.astype(np.uint8)

        # Convertir l'image NumPy en image PIL
        image_pil = Image.fromarray(image_array)

        # S'assurer que l'image est en mode RGB
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        img_width, img_height = image_pil.size

        # Convertir les coordonnées normalisées en pixels avec marge
        x1 = max(0, int(coords['x1'] * img_width) - margin)
        y1 = max(0, int(coords['y1'] * img_height) - margin)
        x2 = min(img_width, int(coords['x2'] * img_width) + margin)
        y2 = min(img_height, int(coords['y2'] * img_height) + margin)

        # Extraire la région
        region = image_pil.crop((x1, y1, x2, y2))

        return region, (x1, y1, x2, y2)

    def calculer_boite_finale(self, boxes_df):
        """Calcule une boîte englobante à partir des coordonnées détectées"""
        if len(boxes_df) == 0:
            return None

        # Trouver les coordonnées min/max pour englober toutes les boîtes
        x1_final = boxes_df['x1'].min()
        y1_final = boxes_df['y1'].min()
        x2_final = boxes_df['x2'].max()
        y2_final = boxes_df['y2'].max()

        return {
            'x1': x1_final,
            'y1': y1_final,
            'x2': x2_final,
            'y2': y2_final
        }

    def extract_boxes_only(self, ocr_result):
        """Extrait uniquement les boîtes des résultats OCR"""
        boxes = []

        for page in ocr_result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        # Obtenir les coordonnées
                        box = word.geometry

                        # Format attendu: ((x1, y1), (x2, y2))
                        if isinstance(box, tuple) and len(box) == 2:
                            (x1, y1), (x2, y2) = box
                            boxes.append([x1, y1, x2, y2])

        # Créer un DataFrame avec les coordonnées
        import pandas as pd
        df = pd.DataFrame({
            'x1': [box[0] for box in boxes],
            'y1': [box[1] for box in boxes],
            'x2': [box[2] for box in boxes],
            'y2': [box[3] for box in boxes]
        })

        return df

    def process_pdf(self, pdf_path):
        """Traite un PDF et extrait le texte structuré"""
        try:
            # Charger le PDF avec PyMuPDF
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                return {"error": "Le PDF ne contient aucune page"}
            
            page = doc.load_page(0)  # Première page
            
            # Obtenir un rendu haute qualité de la page
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Facteur de zoom 2
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            # Si l'image est en CMYK ou a un canal alpha
            if pix.n == 4:
                # Convertir en RGB si nécessaire
                if pix.colorspace == fitz.csRGB:  # RGB+alpha
                    # Ignorer le canal alpha
                    img_array = img_array[:, :, :3]
                else:  # CMYK
                    img_rgb = Image.frombytes("CMYK", [pix.w, pix.h], pix.samples)
                    img_rgb = img_rgb.convert("RGB")
                    img_array = np.array(img_rgb)
            
            # Détection de texte avec doctr
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor, fast_tiny
            
            # Initialiser le modèle OCR pré-entraîné
            det_model = fast_tiny(pretrained=True).to(self.device)
            predictor = ocr_predictor(det_arch=det_model).to(self.device)
            
            # Exécuter l'OCR sur la première page
            doc_for_ocr = DocumentFile.from_pdf(pdf_path)
            first_page_for_ocr = doc_for_ocr[0]  
            first_page_only = [first_page_for_ocr]
            result = predictor(first_page_only)
            
            # Extraire les boîtes
            boxes_df = self.extract_boxes_only(result)
            
            # Définir les coordonnées de la boîte englobante
            coords_boite_englobante = self.calculer_boite_finale(boxes_df)
            
            # Extraire la zone englobante
            zone_englobante, _ = self.extraire_zone_normalisee(
                img_array, coords_boite_englobante, margin=0)
            
            # Redimensionner à une taille cible uniforme
            TARGET_WIDTH = 1500
            TARGET_HEIGHT = 2000
            zone_redimensionnee = zone_englobante.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
            
            # Traiter l'image
            resultats = self.traiter_image_complete(
                image_redimensionnee=zone_redimensionnee,
                visualiser_roi_flag=False
            )
            
            return resultats["structure_constat"] if "structure_constat" in resultats else resultats
            
        except Exception as e:
            return {"error": str(e)}

# Fonction d'utilisation pour traiter un PDF
def process_pdf_file(pdf_path):
    processor = ConstatProcessor()
    results = processor.process_pdf(pdf_path)
    return results