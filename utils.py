import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
from collections import Counter, defaultdict


ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)



ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

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
    return word

def detect_blue_strip(image_path):
    image = cv2.imread(image_path)
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
    word = word.replace("-", "").replace(".", "").replace(" ", "")
    word = limpar_placa(word)

    if len(word) < 7:
        return None

    for pos in [0, 1, 2]:
        if word[pos].isdigit():
            word = change_char_in_position(word, pos)
    
    if is_new_plate and len(word) > 4 and word[4].isdigit():
        word = change_char_in_position(word, 4)

    # Verifica os dois formatos válidos
    if re.match(r'^[A-Z]{3}\d{4}$', word) or re.match(r'^[A-Z]{3}\d[A-Z]\d{2}$', word):
        return word

    return None


def detect_blue_strip_from_array(image):
    if image is None:
        return False

    height, width = image.shape[:2]
    top_strip = image[0:int(height * 0.15), 0:width]

    hsv = cv2.cvtColor(top_strip, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    blue_ratio = cv2.countNonZero(mask) / (top_strip.shape[0] * top_strip.shape[1])
    return blue_ratio > 0.03



def extract_plate_from_image(image, plates):
    is_new_plate = detect_blue_strip_from_array(image)
    result = ocr.ocr(image, cls=True)

    if not result or result[0] is None:
        print("No text detected.")
        return None

    detected_words = []
    for line in result:
        for word_info in line:
            text = word_info[1][0].replace(" ", "")
            if text.lower() == "brasil":
                is_new_plate = True
            else:
                detected_words.append((text, word_info[1][1]))

    # Tenta palavras individuais
    for word_tuple in detected_words:
        corrected = correct_plate(word_tuple[0], is_new_plate)
        if corrected and corrected not in plates:
            return corrected

    # Tenta combinações de palavras
    for i in range(len(detected_words)):
        for j in range(i+1, len(detected_words)):
            combined = detected_words[i][0] + detected_words[j][0]
            corrected = correct_plate(combined, is_new_plate)
            if corrected and corrected not in plates:
                return corrected

    return None




def hamming_distance(a, b):
    if len(a) != len(b):
        return float('inf')
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

def agrupar_placas_por_hamming(plate_counts, max_dist=1):
    grupos = []
    usadas = set()

    placas = list(plate_counts.keys())

    for i in range(len(placas)):
        if placas[i] in usadas:
            continue

        grupo = [placas[i]]
        usadas.add(placas[i])

        for j in range(i + 1, len(placas)):
            if placas[j] not in usadas and hamming_distance(placas[i], placas[j]) <= max_dist:
                grupo.append(placas[j])
                usadas.add(placas[j])

        grupos.append(grupo)

    return grupos
