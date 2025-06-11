# Detector de Placas de Veículos

Este projeto consiste em uma API construída com FastAPI para detectar e reconhecer placas de veículos em imagens ou vídeos. Utiliza modelos YOLO para detecção de placas, PaddleOCR para reconhecimento de caracteres e realiza consultas de dados de placas via WDAPI2 com cache local para otimização de chamadas.

## Funcionalidades

* Detecção e extração de placas em imagens e vídeos
* Reconhecimento de caracteres das placas (OCR) com correções de caracteres comuns
* Agrupamento de placas similares usando distância de Hamming
* Consulta de dados de placas via WDAPI2 com cache local
* Endpoints REST para detecção e consulta de placas

## Pré-requisitos

* Python 3.10 (não testado em outras versões)
* Git
* (Opcional) GPU com CUDA para acelerar inferência do YOLO

## Instalação

1. Clone o repositório:

   ```bash
   git clone <REPOSITORY_URL>
   cd <REPOSITORY_DIRECTORY>
   ```

2. Crie e ative um ambiente virtual:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows
   ```

3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

   Caso não exista `requirements.txt`, instale manualmente:

   ```bash
   pip install fastapi uvicorn python-dotenv requests ultralytics opencv-python paddleocr matplotlib numpy
   ```

## Configuração

1. Crie um arquivo `.env` na raiz do projeto com a variável:

   ```env
   API_TOKEN=<SEU_TOKEN_DA_WDAPI2>
   ```

2. Verifique o caminho do modelo YOLO em `models.py` ou `utils.py` (padrão: `models/last.pt`).

## Como executar

1. Inicie a aplicação FastAPI:

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. Acesse a documentação interativa em [http://localhost:8000/docs](http://localhost:8000/docs)

## Endpoints

### POST /detect-plate/

Detecta placas em uma imagem ou vídeo (payload em base64).

* **Request**:

  ```json
  {
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
  }
  ```

* **Response**:

  ```json
  {
    "message": "ok",
    "plates": {
      "ABC1234": {
        "frequency": 1,
        "frame": 1,
        "image": "<base64 do recorte>",
        "image_resolution": 123456
      }
    }
  }
  ```

### GET /consulta-placa/{placa}

Consulta dados da placa na WDAPI2 (com cache local).

* **Request**:

  ```bash
  GET /consulta-placa/ABC1234
  ```

* **Response** (cache ou API):

  ```json
  {
    "source": "cache",
    "data": { ... }
  }
  ```

## Exemplos com curl

```bash
# Detectar placa
curl -X POST http://localhost:8000/detect-plate/ \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,AAA..."}'

# Consultar placa
curl http://localhost:8000/consulta-placa/ABC1234
```

## Licença

Este projeto está licenciado sob a MIT License.
