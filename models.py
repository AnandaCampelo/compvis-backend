from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

def classify_image(image_path, model=None):
    """
    Classifica uma imagem usando o modelo YOLO.
    Args:
        image_path (str): Caminho da imagem a ser classificada.
        model (YOLO, optional): Modelo YOLO carregado. Se None, o modelo será carregado dentro da função.
    """
    if model is None:
        # Carregar o modelo YOLO se não for passado como argumento
        model = YOLO('last.pt')    

    results = model(image_path)
    
    # Display results
    for result in results:
        img_rgb = cv2.cvtColor(result.plot(show=False), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()

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
        # Carregar o modelo YOLO se não for passado como argumento
        model = YOLO('last.pt')

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

            # Salvar opcionalmente para debug
            output_filename = f'frame_crop_{i}_{j}.png'
            imgnames.append(output_filename)
            cv2.imwrite(output_filename, cropped)

    return cropped_images
