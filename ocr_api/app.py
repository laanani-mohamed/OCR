from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import tempfile
import logging
import time
from processor_image import ImageProcessor
from processor_pdf import ConstatProcessor

# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite à 16 Mo
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()  # Dossier temporaire pour les fichiers

# Extensions permises
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

def allowed_file(filename):
    """Vérifier si le fichier a une extension autorisée"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint pour vérifier que l'API fonctionne"""
    return jsonify({"status": "ok"})

@app.route('/models/check', methods=['GET'])
def check_models():
    """Endpoint pour vérifier que les modèles sont chargés"""
    try:
        # Vérifier PaddleOCR
        from paddleocr import PaddleOCR
        paddle_ocr = PaddleOCR(use_angle_cls=True, lang='fr')
        
        # Vérifier TrOCR
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
        
        # Vérifier doctr
        from doctr.models import ocr_predictor, fast_tiny
        doctr_model = fast_tiny(pretrained=True)
        
        return jsonify({
            "status": "ok",
            "models": {
                "paddleocr": "loaded",
                "trocr": "loaded",
                "doctr": "loaded"
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/ocr/image', methods=['POST'])
def process_image():
    """Endpoint pour traiter une image avec OCR"""
    # Démarrer le chronomètre
    start_time = time.time()
    
    # Vérifier qu'un fichier a été envoyé
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier n'a été envoyé"}), 400
    
    file = request.files['file']
    
    # Vérifier que le fichier a un nom
    if file.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné"}), 400
    
    # Vérifier que c'est un fichier image autorisé
    if file and allowed_file(file.filename) and file.filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']:
        try:
            # Sauvegarder le fichier temporairement
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Initialiser le processeur d'images et traiter l'image
            image_processor = ImageProcessor()
            result = image_processor.process_image(filepath)
            
            # Nettoyer le fichier temporaire
            os.remove(filepath)
            
            # Calculer le temps d'exécution
            execution_time = time.time() - start_time
            
            # Ajouter le temps d'exécution au résultat
            result['execution_time'] = {
                'seconds': execution_time,
            }
            
            # Journaliser le temps d'exécution
            logger.info(f"Image traitée en {execution_time:.2f} secondes: {file.filename}")
            
            return jsonify(result)
        
        except Exception as e:
            # Calculer le temps d'exécution même en cas d'erreur
            execution_time = time.time() - start_time
            logger.error(f"Erreur lors du traitement de l'image ({execution_time:.2f}s): {str(e)}")
            return jsonify({
                "error": f"Erreur lors du traitement: {str(e)}",
                "execution_time": {
                    'seconds': execution_time,
                }
            }), 500
    
    return jsonify({"error": "Type de fichier non pris en charge"}), 400

@app.route('/ocr/pdf', methods=['POST'])
def process_pdf():
    """Endpoint pour traiter un PDF de constat amiable"""
    # Démarrer le chronomètre
    start_time = time.time()
    
    # Vérifier qu'un fichier a été envoyé
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier n'a été envoyé"}), 400
    
    file = request.files['file']
    
    # Vérifier que le fichier a un nom
    if file.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné"}), 400
    
    # Vérifier que c'est un fichier PDF
    if file and allowed_file(file.filename) and file.filename.rsplit('.', 1)[1].lower() == 'pdf':
        try:
            # Sauvegarder le fichier temporairement
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Initialiser le processeur de PDF et traiter le fichier
            pdf_processor = ConstatProcessor()
            result = pdf_processor.process_pdf(filepath)
            
            # Nettoyer le fichier temporaire
            os.remove(filepath)
            
            # Calculer le temps d'exécution
            execution_time = time.time() - start_time
            
            # Si le résultat est un dictionnaire, ajouter le temps d'exécution
            if isinstance(result, dict):
                result['execution_time'] = {
                    'seconds': execution_time,
                }
            else:
                # Sinon, encapsuler le résultat
                result = {
                    'data': result,
                    'execution_time': {
                        'seconds': execution_time,
                    }
                }
            
            # Journaliser le temps d'exécution
            logger.info(f"PDF traité en {execution_time:.2f} secondes: {file.filename}")
            
            return jsonify(result)
        
        except Exception as e:
            # Calculer le temps d'exécution même en cas d'erreur
            execution_time = time.time() - start_time
            logger.error(f"Erreur lors du traitement du PDF ({execution_time:.2f}s): {str(e)}")
            return jsonify({
                "error": f"Erreur lors du traitement: {str(e)}",
                "execution_time": {
                    'seconds': execution_time,
                }
            }), 500
    
    return jsonify({"error": "Type de fichier non pris en charge"}), 400

if __name__ == '__main__':
    debug_mode = False  # Simplifié ici
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Démarrage de l'API OCR sur le port {port}, debug={debug_mode}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)