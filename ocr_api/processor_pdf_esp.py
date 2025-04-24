import os
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path

class PDFProcessor:
    def __init__(self):
        # Vérifier si un GPU est disponible et l'utiliser
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔹 Utilisation du périphérique : {self.device}")
        
        # Charger le modèle TrOCR
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to(self.device)
        
        # Définir les régions d'intérêt pour chaque partie du document
        self.roi_mapping = {
            'vehicule_A': {
                'partie_1': {'x': 0, 'y': 680, 'w': 850, 'h': 350, 'line_height': 50, 'x_offsets': [310, 10, 20, 0, 130, 180, 130]},
                'partie_2': {'x': 0, 'y': 1180, 'w': 450, 'h': 230, 'line_height': 46, 'x_offsets': [0, 0, 0, 0, 0]},
                'partie_3': {'x': 0, 'y': 1470, 'w': 850, 'h': 470, 'line_height': 50, 'x_offsets': [170, 280, 0, 170, 0, 0, 130, 180, 130]},
                'partie_4': {'x': 0, 'y': 2090, 'w': 850, 'h': 540, 'line_height': 50, 'x_offsets': [310, 10, 20, 0, 130, 180, 130, 400, 350, 220]},
                'partie_5': {'x': 0, 'y': 3010, 'w': 550, 'h': 200, 'line_height': 60, 'x_offsets': [0, 0, 0]},
                'partie_6': {'x': 0, 'y': 3243, 'w': 850, 'h': 280, 'line_height': 62, 'x_offsets': [0, 0, 0]}
            },
            'vehicule_B': {
                'partie_1': {'x': 1580, 'y': 680, 'w': 850, 'h': 350, 'line_height': 50, 'x_offsets': [310, 10, 20, 0, 130, 140, 100]},
                'partie_2': {'x': 1580, 'y': 1180, 'w': 450, 'h': 230, 'line_height': 46, 'x_offsets': [0, 0, 0, 0, 0]},
                'partie_3': {'x': 1580, 'y': 1470, 'w': 850, 'h': 470, 'line_height': 50, 'x_offsets': [130, 230, 0, 130, 0, 0, 100, 150, 100]},
                'partie_4': {'x': 1580, 'y': 2080, 'w': 900, 'h': 570, 'line_height': 50, 'x_offsets': [0, 280, 10, 20, 0, 100, 150, 100, 350, 320, 190]},
                'partie_5': {'x': 1940, 'y': 3010, 'w': 550, 'h': 200, 'line_height': 60, 'x_offsets': [0, 0, 0]},
                'partie_6': {'x': 1580, 'y': 3230, 'w': 850, 'h': 250, 'line_height': 62, 'x_offsets': [0, 0, 0]}
            },
            'en_tete': {
                'partie_1': {'x': 0, 'y': 207, 'w': 400, 'h': 80, 'line_height': 80, 'x_offsets': [0]},
                'partie_2': {'x': 570, 'y': 210, 'w': 1350, 'h': 80, 'line_height': 80, 'x_offsets': [0]}
            }
        }
    
    def process_pdf(self, pdf_path):
        """
        Traite un PDF et extrait le texte de chaque région d'intérêt.
        
        Args:
            pdf_path (str): Chemin vers le fichier PDF à traiter
            
        Returns:
            dict: Dictionnaire contenant le texte extrait pour chaque partie du document
        """
        # Convertir le PDF en images
        images = convert_from_path(pdf_path, dpi=300)
        
        # Vérifier si le PDF contient au moins une page
        if len(images) == 0:
            return {"error": "Le PDF ne contient aucune page"}
        
        # Sélectionner uniquement la 1ère page
        image = images[0]
        
        # Convertir l'image PIL en numpy.ndarray
        image_np = np.array(image)
        
        # Initialiser le dictionnaire pour stocker les résultats
        results = {
            'vehicule_A': {},
            'vehicule_B': {},
            'en_tete': {}
        }
        
        # Parcourir chaque section du document
        for section, parties in self.roi_mapping.items():
            for partie_name, roi_info in parties.items():
                # Extraire les paramètres de la région d'intérêt
                x, y, w, h = roi_info['x'], roi_info['y'], roi_info['w'], roi_info['h']
                line_height = roi_info['line_height']
                x_offsets = roi_info['x_offsets']
                
                # Vérifier que les coordonnées sont valides
                if y + h > image_np.shape[0] or x + w > image_np.shape[1]:
                    results[section][partie_name] = {"error": "Coordonnées invalides"}
                    continue
                
                # Extraire la région d'intérêt
                roi = image_np[y:y+h, x:x+w]
                
                # Calculer le nombre de lignes
                num_lines = h // line_height
                
                # S'assurer que la liste x_offsets a la bonne longueur
                if len(x_offsets) < num_lines:
                    x_offsets += [0] * (num_lines - len(x_offsets))
                
                # Initialiser la liste pour stocker le texte de chaque ligne
                lines_text = []
                
                # Traiter chaque ligne
                for i in range(min(num_lines, len(x_offsets))):
                    line_y = i * line_height
                    x_offset = x_offsets[i]
                    adjusted_x = max(0, x_offset)
                    adjusted_w = max(0, w - x_offset)
                    
                    # Vérifier que les dimensions sont valides avant de découper
                    if adjusted_x >= roi.shape[1] or line_y + line_height > roi.shape[0]:
                        lines_text.append(f"Ligne {i+1}: Dimensions invalides")
                        continue
                    
                    # Extraire la ligne
                    line = roi[line_y:line_y + line_height, adjusted_x:adjusted_x + adjusted_w]
                    
                    if line.size == 0:
                        lines_text.append(f"Ligne {i+1}: Vide")
                        continue
                    
                    # Convertir en image PIL
                    line_image = Image.fromarray(line)
                    
                    # Appliquer TrOCR sur la ligne
                    pixel_values = self.processor(line_image, return_tensors="pt").pixel_values.to(self.device)
                    generated_ids = self.model.generate(pixel_values).to("cpu")  # Ramener sur CPU pour décodage
                    text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    lines_text.append(text)
                
                # Stocker les résultats pour cette partie
                results[section][partie_name] = lines_text
        
        return results