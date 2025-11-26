"""
preprocess_opencv.py - Preprocesamiento de imágenes con OpenCV

Pipeline completo de preprocesamiento para optimizar la calidad
de reconocimiento OCR:
- Conversión a escala de grises
- Binarización (Otsu + adaptativa)
- Corrección de inclinación (deskew)
- Limpieza de ruido
- Normalización de tamaño

Uso:
    from llarri.inference.preprocess_opencv import preprocess_for_ocr
    
    tensor = preprocess_for_ocr("image.jpg")
    # o
    tensor = preprocess_for_ocr(pil_image)
"""

import numpy as np
from typing import Union, Optional, Tuple
from pathlib import Path

# OpenCV es opcional para no forzar dependencia
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from PIL import Image


def _ensure_cv2():
    """Verifica que OpenCV esté disponible."""
    if not HAS_CV2:
        raise ImportError(
            "OpenCV no está instalado. Instala con: pip install opencv-python-headless"
        )


def load_image(image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Carga imagen desde múltiples fuentes a formato numpy BGR.
    
    Args:
        image_input: Path, PIL Image, o numpy array
        
    Returns:
        Imagen como numpy array BGR
    """
    _ensure_cv2()
    
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_input}")
        return img
    
    elif isinstance(image_input, Image.Image):
        # PIL a OpenCV (RGB a BGR)
        img = np.array(image_input)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    elif isinstance(image_input, np.ndarray):
        return image_input.copy()
    
    else:
        raise TypeError(f"Tipo de imagen no soportado: {type(image_input)}")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convierte imagen a escala de grises."""
    _ensure_cv2()
    
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Shape de imagen inválido: {image.shape}")


def binarize_otsu(image: np.ndarray) -> np.ndarray:
    """
    Binarización usando método de Otsu.
    Bueno para imágenes con iluminación uniforme.
    """
    _ensure_cv2()
    
    gray = to_grayscale(image)
    
    # Aplicar Gaussian blur para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's thresholding
    _, binary = cv2.threshold(
        blurred, 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return binary


def binarize_adaptive(
    image: np.ndarray,
    block_size: int = 31,
    c: int = 10,
) -> np.ndarray:
    """
    Binarización adaptativa.
    Mejor para imágenes con iluminación no uniforme.
    
    Args:
        image: Imagen de entrada
        block_size: Tamaño del bloque para calcular threshold (debe ser impar)
        c: Constante a restar del mean/gaussian
    """
    _ensure_cv2()
    
    gray = to_grayscale(image)
    
    # Asegurar que block_size sea impar
    if block_size % 2 == 0:
        block_size += 1
    
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c
    )
    
    return binary


def binarize_combined(image: np.ndarray) -> np.ndarray:
    """
    Binarización combinada: Otsu + Adaptativa.
    Usa la que dé mejor contraste.
    """
    _ensure_cv2()
    
    otsu = binarize_otsu(image)
    adaptive = binarize_adaptive(image)
    
    # Calcular contraste (desviación estándar)
    otsu_std = np.std(otsu)
    adaptive_std = np.std(adaptive)
    
    # Usar la que tenga mejor contraste
    return otsu if otsu_std > adaptive_std else adaptive


def remove_noise(
    image: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Elimina ruido usando operaciones morfológicas.
    
    Args:
        image: Imagen binaria
        kernel_size: Tamaño del kernel morfológico
        iterations: Número de iteraciones
    """
    _ensure_cv2()
    
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    
    # Opening (erosion + dilation) para eliminar puntos pequeños
    cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    # Closing (dilation + erosion) para cerrar gaps pequeños
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    return cleaned


def calculate_skew_angle(image: np.ndarray) -> float:
    """
    Calcula el ángulo de inclinación del texto.
    
    Returns:
        Ángulo en grados (negativo = inclinado a la derecha)
    """
    _ensure_cv2()
    
    gray = to_grayscale(image)
    
    # Detectar bordes
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detectar líneas con Hough
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=10
    )
    
    if lines is None or len(lines) == 0:
        return 0.0
    
    # Calcular ángulos de las líneas
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Solo considerar líneas casi horizontales
        if abs(angle) < 45:
            angles.append(angle)
    
    if not angles:
        return 0.0
    
    # Retornar mediana de ángulos
    return float(np.median(angles))


def deskew(image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
    """
    Corrige la inclinación del texto.
    
    Args:
        image: Imagen de entrada
        angle: Ángulo a corregir (si None, se calcula automáticamente)
    """
    _ensure_cv2()
    
    if angle is None:
        angle = calculate_skew_angle(image)
    
    # Si el ángulo es muy pequeño, no rotar
    if abs(angle) < 0.5:
        return image
    
    # Obtener dimensiones
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Crear matriz de rotación
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calcular nuevas dimensiones para no recortar
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Ajustar matriz para centrar
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Aplicar rotación
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated


def resize_maintaining_aspect(
    image: np.ndarray,
    target_height: int = 64,
    max_width: Optional[int] = None,
) -> np.ndarray:
    """
    Redimensiona manteniendo aspect ratio.
    
    Args:
        image: Imagen de entrada
        target_height: Altura objetivo
        max_width: Ancho máximo (si None, no limita)
    """
    _ensure_cv2()
    
    h, w = image.shape[:2]
    
    # Calcular nuevo ancho manteniendo aspect ratio
    scale = target_height / h
    new_width = int(w * scale)
    
    # Limitar ancho máximo si se especifica
    if max_width and new_width > max_width:
        new_width = max_width
    
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
    
    return resized


def pad_to_size(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    pad_value: int = 255,
) -> np.ndarray:
    """
    Añade padding para alcanzar tamaño objetivo.
    
    Args:
        image: Imagen de entrada
        target_width: Ancho objetivo
        target_height: Altura objetivo  
        pad_value: Valor de padding (255 = blanco)
    """
    h, w = image.shape[:2]
    
    # Calcular padding necesario
    pad_h = max(0, target_height - h)
    pad_w = max(0, target_width - w)
    
    # Padding simétrico
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    if len(image.shape) == 2:
        padded = np.full((target_height, target_width), pad_value, dtype=image.dtype)
    else:
        padded = np.full((target_height, target_width, image.shape[2]), pad_value, dtype=image.dtype)
    
    # Colocar imagen centrada
    y_start = top
    y_end = top + h
    x_start = left
    x_end = left + w
    
    # Recortar si es más grande
    if h > target_height:
        y_start = 0
        y_end = target_height
        img_y_start = (h - target_height) // 2
        image = image[img_y_start:img_y_start + target_height]
        h = target_height
    
    if w > target_width:
        x_start = 0
        x_end = target_width
        img_x_start = (w - target_width) // 2
        image = image[:, img_x_start:img_x_start + target_width]
        w = target_width
    
    padded[y_start:y_start+h, x_start:x_start+w] = image[:h, :w]
    
    return padded


def normalize_for_model(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Normaliza imagen para entrada al modelo.
    
    Args:
        image: Imagen preprocesada
        target_size: (height, width) objetivo
        
    Returns:
        Array normalizado [0, 1] con shape (C, H, W)
    """
    target_h, target_w = target_size
    
    # Asegurar 3 canales
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Redimensionar si es necesario
    h, w = image.shape[:2]
    if h != target_h or w != target_w:
        if HAS_CV2:
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            # Fallback con PIL
            pil_img = Image.fromarray(image)
            pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            image = np.array(pil_img)
    
    # Normalizar a [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))
    
    return image


def preprocess_for_ocr(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    binarize: bool = True,
    deskew_image: bool = True,
    remove_noise_flag: bool = True,
    return_intermediate: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Pipeline completo de preprocesamiento para OCR.
    
    Args:
        image_input: Imagen de entrada (path, PIL, numpy)
        target_size: Tamaño de salida (height, width)
        binarize: Aplicar binarización
        deskew_image: Corregir inclinación
        remove_noise_flag: Eliminar ruido
        return_intermediate: Retornar pasos intermedios
        
    Returns:
        Tensor normalizado listo para el modelo [C, H, W]
        Si return_intermediate=True, también retorna dict con pasos intermedios
    """
    intermediates = {}
    
    # 1. Cargar imagen
    image = load_image(image_input)
    intermediates['original'] = image.copy()
    
    # 2. Escala de grises
    gray = to_grayscale(image)
    intermediates['grayscale'] = gray.copy()
    
    # 3. Binarización
    if binarize:
        binary = binarize_combined(gray)
        intermediates['binary'] = binary.copy()
    else:
        binary = gray
    
    # 4. Corrección de inclinación
    if deskew_image:
        angle = calculate_skew_angle(binary)
        deskewed = deskew(binary, angle)
        intermediates['deskewed'] = deskewed.copy()
        intermediates['skew_angle'] = angle
    else:
        deskewed = binary
    
    # 5. Eliminación de ruido
    if remove_noise_flag:
        cleaned = remove_noise(deskewed)
        intermediates['cleaned'] = cleaned.copy()
    else:
        cleaned = deskewed
    
    # 6. Normalización para modelo
    normalized = normalize_for_model(cleaned, target_size)
    intermediates['normalized_shape'] = normalized.shape
    
    if return_intermediate:
        return normalized, intermediates
    
    return normalized


def preprocess_batch(
    images: list,
    target_size: Tuple[int, int] = (224, 224),
    **kwargs,
) -> np.ndarray:
    """
    Preprocesa un batch de imágenes.
    
    Args:
        images: Lista de imágenes (paths, PIL, numpy)
        target_size: Tamaño objetivo
        **kwargs: Argumentos adicionales para preprocess_for_ocr
        
    Returns:
        Batch de tensores [B, C, H, W]
    """
    processed = []
    
    for img in images:
        tensor = preprocess_for_ocr(img, target_size, **kwargs)
        processed.append(tensor)
    
    return np.stack(processed, axis=0)


# Función de conveniencia para PyTorch
def preprocess_to_tensor(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    device: str = 'cpu',
    **kwargs,
):
    """
    Preprocesa y convierte directamente a tensor PyTorch.
    
    Returns:
        torch.Tensor con shape [1, C, H, W]
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch no instalado. Usa preprocess_for_ocr() para numpy.")
    
    numpy_tensor = preprocess_for_ocr(image_input, target_size, **kwargs)
    tensor = torch.from_numpy(numpy_tensor).unsqueeze(0)
    
    return tensor.to(device)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocesar imagen para OCR")
    parser.add_argument('image', type=str, help='Ruta a la imagen')
    parser.add_argument('--output', '-o', type=str, default=None, help='Guardar resultado')
    parser.add_argument('--show', action='store_true', help='Mostrar pasos intermedios')
    parser.add_argument('--no-binarize', action='store_true', help='No binarizar')
    parser.add_argument('--no-deskew', action='store_true', help='No corregir inclinación')
    
    args = parser.parse_args()
    
    result, intermediates = preprocess_for_ocr(
        args.image,
        binarize=not args.no_binarize,
        deskew_image=not args.no_deskew,
        return_intermediate=True,
    )
    
    print(f"Input: {args.image}")
    print(f"Output shape: {result.shape}")
    print(f"Skew angle: {intermediates.get('skew_angle', 'N/A')}°")
    
    if args.output:
        # Guardar versión procesada (antes de normalizar)
        output_img = intermediates.get('cleaned', intermediates.get('binary'))
        cv2.imwrite(args.output, output_img)
        print(f"Saved to: {args.output}")
    
    if args.show and HAS_CV2:
        for name, img in intermediates.items():
            if isinstance(img, np.ndarray) and len(img.shape) >= 2:
                cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

