from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64, tempfile, os, json,utils,requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# -------------- CORS --------------
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],            # <- or your exact frontend URL
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# -------------- Cache utils --------------
API_TOKEN  = os.getenv("API_TOKEN","default_token")
API_URL    = "https://wdapi2.com.br/consulta/{placa}/{token}"
CACHE_FILE = "cache.json"

def load_cache():
    """
    Carrega o cache de placa-resposta do arquivo JSON.
    Retorna um dicionário vazio se o arquivo não existir ou estiver inválido.
    """
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # Se o arquivo estiver corrompido, ignora e recria
            return {}
    return {}


def save_cache(cache: dict):
    """
    Salva o dicionário de cache em disco no formato JSON.
    """
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=4)
    except IOError as e:
        # Falha ao salvar; pode logar se necessário
        print(f"Erro ao salvar cache: {e}")



# -------------- Request model --------------
class ImageRequest(BaseModel):
    image: str  # data URL or pure base64

# -------------- OCR endpoint --------------
@app.post("/detect-plate/", response_model=dict)
async def detect_plate(req: ImageRequest = Body(...)):
    try:
        # strip optional "data:image/...;base64," header
        header, _, b64 = req.image.partition(",")
        raw = base64.b64decode(b64 or req.image)
    except Exception:
        raise HTTPException(400, "Invalid base64 payload")

    # write temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
        tf.write(raw)
        path = tf.name

    try:
        plates = utils.analyze_video(path)
    finally:
        os.unlink(path)

    if not plates:
        # positional-correct:
        return JSONResponse({"message": "No plates detected."}, 404)

    return {"message": "ok", "plates": plates}

# -------------- Consulta endpoint (unchanged) --------------
@app.get("/consulta-placa/{placa}")
async def consulta_placa(placa: str):
    """
    Consulta os dados de uma placa usando a WDAPI2,
    armazenando em cache local para economizar tokens.

    Args:
        placa (str): Número da placa a ser consultada.
    Returns:
        JSONResponse: Dados da placa, vindos do cache ou da API externa.
    """
    placa = placa.upper()
    cache = load_cache()

    # Se já estiver em cache, retorna imediatamente
    if placa in cache:
        return JSONResponse(status_code=200, content={"source": "cache", "data": cache[placa]})

    # Caso contrário, faz a chamada externa
    url = API_URL.format(placa=placa, token=API_TOKEN)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Armazena no cache local
        cache[placa] = data
        save_cache(cache)

        return JSONResponse(status_code=200, content={"source": "api", "data": data})

    except requests.RequestException as e:
        # Em caso de erro na chamada externa
        return JSONResponse(status_code=502, content={"message": "Erro ao consultar API externa.", "details": str(e)})
