from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
import shutil
import os
import requests

from detect import get_detected_plates # detect.py

app = FastAPI()

API_TOKEN = os.getenv("API_TOKEN", "default_token_value")
API_URL = "https://wdapi2.com.br/consulta/{placa}/{token}"

@app.post("/detect-plate/")
async def detect_plate(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        # 1. Detecta placas com lógica externa
        plates = get_detected_plates(temp_path)
        if not plates:
            return JSONResponse(content={"error": "Nenhuma placa detectada"}, status_code=404)

        # 2. Faz a consulta na API para cada placa
        respostas = []
        for placa in plates:
            try:
                url = API_URL.format(placa=placa, token=API_TOKEN)
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    respostas.append(response.json())
                else:
                    respostas.append({"placa": placa, "erro": f"status {response.status_code}"})
            except Exception as e:
                respostas.append({"placa": placa, "erro": str(e)})

        return JSONResponse(content=respostas, status_code=200)

    finally:
        os.remove(temp_path)