import os
import requests
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
from collections import defaultdict
import re
import base64

debug = True

# Aux Functions

# ================== Classify and Crop ==================
def classify_and_crop(image, model=None):
    """
    Detecta objetos em uma imagem e recorta as placas detectadas.
    Args:
        image (numpy.ndarray): Imagem de entrada.
        model (YOLO, optional): Modelo YOLO carregado. Se None, o modelo será carregado dentro da função.
    Returns:
        list: Lista de imagens recortadas.
    """
    if model is None:
        model = YOLO('models/last.pt')

    # Run inference diretamente com o frame
    results = model(image)

    imgnames = []
    cropped_images = []

    for i, result in enumerate(results):
        boxes = result.boxes.xyxy.cpu().numpy()

        for j, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cropped = image[y1:y2, x1:x2]
            cropped_images.append(cropped)

            # if debug:
            #     output_filename = f'frame_crop_{i}_{j}.png'
            #     imgnames.append(output_filename)
            #     cv2.imwrite(output_filename, cropped)

    return cropped_images

# ================== Classify Text ==================

def change_char_in_position(word, position):
    if position < len(word):
        if word[position].isdigit():
            digit = word[position]
            if digit == '8':
                word = word[:position] + 'B' + word[position+1:]
            elif digit == '1':
                word = word[:position] + 'I' + word[position+1:]
            elif digit == '0':
                word = word[:position] + 'O' + word[position+1:]
            elif digit == '5':
                word = word[:position] + 'S' + word[position+1:]
            elif digit == '6':
                word = word[:position] + 'G' + word[position+1:]
            elif digit == '3':
                word = word[:position] + 'J' + word[position+1:]
    return word

def change_number_in_position(word, position):
    if position < len(word):
        if word[position].isalpha():
            letter = word[position]
            if letter == 'B':
                word = word[:position] + '8' + word[position+1:]
            elif letter == 'I':
                word = word[:position] + '1' + word[position+1:]
            elif letter == 'O':
                word = word[:position] + '0' + word[position+1:]
            elif letter == 'S':
                word = word[:position] + '5' + word[position+1:]
            elif letter == 'G':
                word = word[:position] + '6' + word[position+1:]
            elif letter == 'J':
                word = word[:position] + '3' + word[position+1:]
    return word

def detect_blue_strip(image):
    if image is None:
        return False
    
    height, width = image.shape[:2]
    top_strip = image[0:int(height * 0.25), 0:width]  # faixa de cima

    hsv = cv2.cvtColor(top_strip, cv2.COLOR_BGR2HSV)

    # Faixa de azul (com tolerância maior para placas borradas)
    lower_blue = np.array([110, 160, 65])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    blue_ratio = cv2.countNonZero(mask) / (top_strip.shape[0] * top_strip.shape[1])
    return blue_ratio > 0.03

def limpar_placa(texto):
    """Remove qualquer caractere que não seja letra ou número."""
    return re.sub(r'[^A-Z0-9]', '', texto.upper())

def correct_plate(word, is_new_plate):
    if len(word) < 7:
        return None
    elif len(word) == 8:
        # Se a placa tiver 8 caracteres, remover o primeiro
        word = word[1:]
    
    char_positions = [0,1,2]
    number_positions = [3,5,6]

    for pos in char_positions:
        if word[pos].isdigit():
            word = change_char_in_position(word, pos)
            
    for pos in number_positions:
        if word[pos].isalpha():
            word = change_number_in_position(word, pos)
    
    if is_new_plate and len(word) > 4 and word[4].isdigit():
        word = change_char_in_position(word, 4)

    # Verifica os dois formatos válidos
    if re.match(r'^[A-Z]{3}\d{4}$', word) or re.match(r'^[A-Z]{3}\d[A-Z]\d{2}$', word):
        return word

    return None

ocr = PaddleOCR(
        use_angle_cls=True, 
        lang='en',
        show_log=False,
    )

def extract_plate_from_image(image_path):
    # Preprocess the image
    is_new_plate = detect_blue_strip(image_path)
    result = ocr.ocr(image_path, cls=True)
        
    if not result or result[0] is None:
        print("Nenhum resultado encontrado.")
        return None    
    
    detected_words = []
    for line in result:
        for word_info in line:
            text = word_info[1][0].replace(" ", "")
            text = limpar_placa(text)
            print(f"Texto detectado: {text}")
            if text.lower() == "brasil":
                is_new_plate = True
            else:
                detected_words.append((text, word_info[1][1]))  # (text, confidence)
                    
    print(f"Palavras detectadas: {detected_words}") 
    
    # Tenta cada palavra isolada
    for word_tuple in detected_words:
        corrected = correct_plate(word_tuple[0], is_new_plate)
        if corrected:
            return corrected

    # Tenta combinações de palavras
    for i in range(len(detected_words)):
        for j in range(i+1, len(detected_words)):
            combined = detected_words[i][0] + detected_words[j][0]
            print(f"Combinando: {combined}")
            corrected = correct_plate(combined, is_new_plate)
            if corrected:
                return corrected
            
    return

# ==================== Hamming Distance ====================

# After collecting the plates, we can use Hamming distance to compare them.
def hamming_distance(s1, s2):
    """
    Calculate the Hamming distance between two strings.
    
    Args:
        s1 (str): First string.
        s2 (str): Second string.
        
    Returns:
        int: The Hamming distance between the two strings.
    """
    if len(s1) != len(s2):
        raise ValueError("Strings must be of the same length")
    
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

def agrupar_placas_por_hamming_completo(placas, max_dist=1):
    """
    Agrupa placas com base na distância de Hamming.
    Args:
        plate_counts (dict): Dicionário com placas e suas contagens.
        max_dist (int): Distância máxima de Hamming para considerar duas placas como pertencentes ao mesmo grupo.
    Returns:
        list: Lista de grupos de placas, onde cada grupo é uma lista de placas.
    """
    grafo = defaultdict(list)

    for i in range(len(placas)):
        for j in range(i + 1, len(placas)):
            if hamming_distance(placas[i], placas[j]) <= max_dist:
                grafo[placas[i]].append(placas[j])
                grafo[placas[j]].append(placas[i])

    visitado = set()
    grupos = []
    
    # Depth-First Search (DFS) para encontrar grupos conectados
    # A ideia é buscar as placas conectadas no grafo.
    # Ao achar uma placa não visitada chama novamente o dfs para buscar as placas conectadas a ela.
    def dfs(placa, grupo):
        visitado.add(placa)
        grupo.append(placa)
        for vizinho in grafo[placa]:
            if vizinho not in visitado:
                dfs(vizinho, grupo)

    for placa in placas:
        if placa not in visitado:
            grupo = []
            dfs(placa, grupo)
            grupos.append(grupo)

    return grupos
    

# ================== Main Function ==================
# Main Function: Analyze the video and call other functions to detect plates and classify them.
def analyze_video(video_path):
    """
    Analyzes a video file to detect license plates and classify the text using OCR.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        list: A list of dictionaries containing the detected plate text and the corresponding frame number.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist.")
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file {video_path}.")
        return
    
    # get the video Frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
        
    success, frame = cap.read()
    frame_number = 0
    
    plates = {}
    
    while success:
        frame_number += 1
        
        # For each second of the video, we will process 3 frames
        if frame_number % int(fps / 3) == 0:
            print(f"Processing frame {frame_number}...")
            
            # 1. Crop the frame to detect plates
            cropped_images = classify_and_crop(frame)
            if not cropped_images:
                print(f"No plates detected in frame {frame_number}.")
                success, frame = cap.read()
                continue
            
            # 2. For each cropped image, classify the text
            for cropped_image in cropped_images:
                plate = extract_plate_from_image(cropped_image)
                
                if plate and plate not in plates:
                    # Encode the cropped image to base64
                    encoded_image = cv2.imencode('.png', cropped_image)[1]
                    data = np.array(encoded_image)
                    data = base64.b64encode(data).decode('utf-8')

                    plates[plate] = {
                        'frequency': 1,
                        'frame': frame_number,
                        'image': data
                    }
                elif plate:
                    plates[plate]['frequency'] += 1
            
        if frame is not None and debug:
            debug_frame = cv2.resize(frame, (720, 480))
            cv2.imshow("Frame", debug_frame)
            key = cv2.waitKey(10)
            if key == ord('q') or not cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE):
                print("Exiting...\n")
                cap.release()
                break
        
        success, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()
    
    # Group plates by Hamming distance
    placas = list(plates.keys())
    grupos = agrupar_placas_por_hamming_completo(placas, max_dist=1)
    
    # Create a new dictionary to store grouped plates
    grouped_plates = {}
    for grupo in grupos:
        if len(grupo) == 1:
            plate = grupo[0]
            grouped_plates[plate] = plates[plate]
        else:
            # If there are multiple plates in the group, take the one with the highest frequency
            max_plate = max(grupo, key=lambda p: plates[p]['frequency'])
            all_group_plates = [plates[p] for p in grupo]
            grouped_plates[max_plate] = {
                'frequency': sum(plates[p]['frequency'] for p in grupo),
                'frame': all_group_plates[0]['frame'],  # Use the frame of the first plate
                'image': plates[max_plate]['image']
            }
    return grouped_plates