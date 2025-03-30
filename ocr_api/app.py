from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import tempfile
from PIL import Image
import io

# Importer nos modules de traitement
from processor_pdf import PDFProcessor
from processor_image import ImageProcessor

# Initialiser FastAPI
app = FastAPI(title="OCR API", description="API pour OCR de documents avec PaddleOCR et TrOCR")

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialiser les processeurs
processor_pdf = PDFProcessor()
processor_image = ImageProcessor()

# Route pour traiter les images
@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    # Créer un ID unique pour ce traitement
    process_id = str(uuid.uuid4())
    
    # Créer un répertoire temporaire pour l'image
    os.makedirs(f"/data/input/{process_id}", exist_ok=True)
    os.makedirs(f"/data/output/{process_id}", exist_ok=True)
    
    # Enregistrer l'image localement
    temp_file_path = f"/data/input/{process_id}/{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        # Traiter l'image avec notre processeur d'image
        results = processor_image.process_image(temp_file_path)
        
        # Ajouter l'ID du processus à la réponse
        response = {
            "process_id": process_id,
            **results
        }
        
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur lors du traitement de l'image: {str(e)}"}
        )

# Route pour traiter les PDF
@app.post("/process_pdf/")
async def process_pdf(file: UploadFile = File(...)):
    # Créer un ID unique pour ce traitement
    process_id = str(uuid.uuid4())
    
    # Créer un répertoire temporaire pour le PDF
    os.makedirs(f"/data/input/{process_id}", exist_ok=True)
    os.makedirs(f"/data/output/{process_id}", exist_ok=True)
    
    # Enregistrer le PDF localement
    temp_file_path = f"/data/input/{process_id}/{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())
    
    # Traiter le PDF avec notre processeur
    try:
        results = processor_pdf.process_pdf(temp_file_path)
        
        # Préparer la réponse
        response = {
            "process_id": process_id,
            "pdf_results": results
        }
        
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur lors du traitement du PDF: {str(e)}"}
        )

@app.get("/")
async def root():
    return {"message": "API OCR opérationnelle. Utilisez /process_image/ pour traiter des images ou /process_pdf/ pour traiter des PDF."}