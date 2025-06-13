# Detector de Placas de Veículos

Este projeto consiste em uma API construída com FastAPI para detectar e reconhecer placas de veículos em imagens ou vídeos. Utiliza modelos YOLOv11 para detecção de placas, PaddleOCR para reconhecimento de caracteres e realiza consultas de dados de placas via WDAPI2 com cache local para otimização de chamadas.

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

## Considerações:

Esse backend foi desenvolvido para analisar placas de veículos no contexto brasileiro, utilizando YOLOv11 e PaddleOCR. A detecção é feita com base em imagens ou vídeos, e o reconhecimento de caracteres é otimizado para lidar com erros comuns de OCR. O cache local melhora a performance das consultas à WDAPI2.

Conforme solicitado pelo professor, embora o código tenha sido desenvolvido para tratar placas brasileiras, é possível alterar o método: `correct_plate` em `utils.py` para adaptar o reconhecimento de caracteres a outros padrões de placas. Ademais, seria necessário alterar a API usada no endpoint `/consulta-placa/{placa}` para uma que forneça dados de placas do país desejado.

## Notas para o professor corretor:

O código acima foi feito o deploy na conta da AWS do Insper e pode ser utilizado através do link: [placas.fernandoa.dev](https://placas.fernandoa.dev). A API está rodando em uma instância EC2 micro de modo a ser acessível publicamente.

Por ser uma instância pequena, o tempo de resposta pode ser muito maior do que o esperado, conforme conversamos por email. Os alunos testaram com imagens que levam cerca de 1 minuto para serem processadas e retornadas, e vídeos de 1 minuto que levaram cerca de 25 minutos para serem processados.

Para realizar testes em vídeos maiores, recomendo hospedar a API localmente, junto com o FrontEnd, que está disponível no repositório [frontend-placas](https://github.com/devfernandoa/CompvisFrontend), a instruções de instalação e execução do FrontEnd estão disponíveis no README desse repositório.


## Licença

Este projeto está licenciado sob a MIT License.
