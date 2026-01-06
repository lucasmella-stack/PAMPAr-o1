/**
 * Servicio de preprocesamiento de imágenes con OpenCV
 * Mejora la calidad de imágenes antes del OCR
 */

import Jimp from 'jimp';

/**
 * Preprocesa una imagen para mejorar el OCR
 * @param {Buffer} imageBuffer - Buffer de la imagen original
 * @returns {Promise<Buffer>} - Buffer de la imagen procesada
 */
export async function preprocessImage(imageBuffer) {
  try {
    console.log('🔧 Preprocessing image for better OCR...');
    
    // Cargar imagen con Jimp
    const image = await Jimp.read(imageBuffer);
    
    // 1. Convertir a escala de grises
    image.grayscale();
    
    // 2. Aumentar contraste
    image.contrast(0.3);
    
    // 3. Normalizar brillo
    image.normalize();
    
    // 4. Aplicar un ligero sharpening para mejorar bordes de texto
    image.convolute([
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0]
    ]);
    
    // 5. Escalar si es muy pequeña (mínimo 1000px de ancho)
    if (image.getWidth() < 1000) {
      const scale = 1000 / image.getWidth();
      image.scale(scale);
    }
    
    // 6. Aplicar threshold adaptativo (binarización)
    // Esto ayuda mucho con texto en fondos irregulares
    image.scan(0, 0, image.bitmap.width, image.bitmap.height, function(x, y, idx) {
      const gray = this.bitmap.data[idx]; // Ya está en escala de grises
      // Threshold simple
      const threshold = 128;
      const newValue = gray > threshold ? 255 : 0;
      this.bitmap.data[idx] = newValue;
      this.bitmap.data[idx + 1] = newValue;
      this.bitmap.data[idx + 2] = newValue;
    });
    
    // Obtener buffer procesado
    const processedBuffer = await image.getBufferAsync(Jimp.MIME_PNG);
    
    console.log(`✅ Image preprocessed: ${imageBuffer.length} -> ${processedBuffer.length} bytes`);
    
    return processedBuffer;
  } catch (error) {
    console.error('⚠️ Error preprocessing image, using original:', error.message);
    return imageBuffer; // Retornar original si falla
  }
}

/**
 * Preprocesamiento avanzado con OpenCV (requiere opencv4nodejs)
 * Este es más potente pero requiere instalación adicional
 */
export async function preprocessImageAdvanced(imageBuffer) {
  try {
    // Intentar cargar opencv4nodejs
    const cv = await import('opencv4nodejs').catch(() => null);
    
    if (!cv) {
      console.log('⚠️ opencv4nodejs not available, using basic preprocessing');
      return preprocessImage(imageBuffer);
    }
    
    console.log('🔧 Advanced preprocessing with OpenCV...');
    
    // Decodificar imagen
    const img = cv.imdecode(imageBuffer);
    
    // 1. Convertir a escala de grises
    const gray = img.cvtColor(cv.COLOR_BGR2GRAY);
    
    // 2. Aplicar filtro bilateral (reduce ruido preservando bordes)
    const bilateral = gray.bilateralFilter(9, 75, 75);
    
    // 3. Threshold adaptativo de Gaussian
    const thresh = bilateral.adaptiveThreshold(
      255,
      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv.THRESH_BINARY,
      11,
      2
    );
    
    // 4. Operaciones morfológicas para limpiar
    const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2, 2));
    const morph = thresh.morphologyEx(kernel, cv.MORPH_CLOSE);
    
    // 5. Deskew (corregir inclinación)
    const deskewed = deskewImage(morph, cv);
    
    // Codificar de vuelta a buffer
    const processedBuffer = cv.imencode('.png', deskewed);
    
    console.log(`✅ Advanced preprocessing complete: ${imageBuffer.length} -> ${processedBuffer.length} bytes`);
    
    return processedBuffer;
  } catch (error) {
    console.error('⚠️ OpenCV error, falling back to basic:', error.message);
    return preprocessImage(imageBuffer);
  }
}

/**
 * Corrige la inclinación de una imagen
 */
function deskewImage(img, cv) {
  try {
    // Detectar líneas con Hough
    const lines = img.houghLinesP(1, Math.PI / 180, 100, 100, 10);
    
    if (lines.length === 0) return img;
    
    // Calcular ángulo promedio
    let totalAngle = 0;
    for (const line of lines) {
      const angle = Math.atan2(line.y2 - line.y1, line.x2 - line.x1) * 180 / Math.PI;
      totalAngle += angle;
    }
    const avgAngle = totalAngle / lines.length;
    
    // Solo corregir si el ángulo es pequeño (< 10 grados)
    if (Math.abs(avgAngle) > 10) return img;
    
    // Rotar imagen
    const center = new cv.Point2(img.cols / 2, img.rows / 2);
    const rotMatrix = cv.getRotationMatrix2D(center, avgAngle, 1.0);
    const rotated = img.warpAffine(rotMatrix, new cv.Size(img.cols, img.rows));
    
    return rotated;
  } catch (error) {
    return img; // Retornar original si falla
  }
}

/**
 * Detecta y recorta el área del documento
 */
export async function detectAndCropDocument(imageBuffer) {
  try {
    const cv = await import('opencv4nodejs').catch(() => null);
    
    if (!cv) {
      return imageBuffer;
    }
    
    const img = cv.imdecode(imageBuffer);
    const gray = img.cvtColor(cv.COLOR_BGR2GRAY);
    
    // Detectar bordes
    const edges = gray.canny(50, 150);
    
    // Encontrar contornos
    const contours = edges.findContours(cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    
    if (contours.length === 0) return imageBuffer;
    
    // Encontrar el contorno más grande (probablemente el documento)
    let maxArea = 0;
    let maxContour = null;
    
    for (const contour of contours) {
      const area = contour.area;
      if (area > maxArea) {
        maxArea = area;
        maxContour = contour;
      }
    }
    
    if (!maxContour || maxArea < img.cols * img.rows * 0.1) {
      return imageBuffer;
    }
    
    // Obtener bounding box y recortar
    const rect = maxContour.boundingRect();
    const cropped = img.getRegion(rect);
    
    return cv.imencode('.png', cropped);
  } catch (error) {
    return imageBuffer;
  }
}

export default {
  preprocessImage,
  preprocessImageAdvanced,
  detectAndCropDocument,
};