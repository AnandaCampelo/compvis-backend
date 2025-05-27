from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import utils
import tempfile
import shutil
import os
import requests

app = FastAPI()

API_TOKEN = os.getenv("API_TOKEN", "default_token_value")
API_URL = "https://wdapi2.com.br/consulta/{placa}/{token}"

# First part: Detect the plates and classify the text using OCR
@app.post("/detect-plate/")
async def detect_plate(file: UploadFile = File(...)):
    """
    Detecta placas em uma imagem e retorna o texto contido na placa.
    
    Args:
        file (UploadFile): Arquivo de imagem enviado pelo usuário.
    Returns:
        JSONResponse: JSON com o texto da placa e uma lista das imagens detectadas codificadas em base64.
    """
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        # Analyze the video file to detect license plates
        results = utils.analyze_video(temp_path)
        if not results:
            return JSONResponse(status_code=404, content={"message": "No plates detected."})
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Plates detected successfully.",
                "plates": results
            }
        )

    finally:
        os.remove(temp_path)