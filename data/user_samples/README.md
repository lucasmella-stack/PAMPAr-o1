# User Samples - Tus Propios Datos

Esta carpeta está diseñada para que puedas agregar tus propias imágenes de texto manuscrito para OCR y fine-tuning.

## Estructura

```
user_samples/
├── images/           # Tus imágenes de texto (.png, .jpg)
├── annotations.txt   # Anotaciones: imagen<TAB>texto
└── README.md         # Este archivo
```

## Cómo usar

### 1. Para Inferencia (OCR sin entrenar)

Simplemente pon tus imágenes en `images/` y ejecuta:

```bash
cd llarri-01
source venv/bin/activate
python scripts/demo_inference.py --folder data/user_samples/images/
```

### 2. Para Fine-Tuning (entrenar con tus datos)

1. **Prepara tus imágenes**:
   - Pon las imágenes en `images/`
   - Formato recomendado: PNG o JPG
   - Idealmente imágenes de líneas de texto (no páginas completas)

2. **Crea el archivo de anotaciones** `annotations.txt`:
   ```
   imagen1.png	Este es el texto de la imagen 1
   imagen2.png	Otro texto manuscrito
   imagen3.jpg	Más ejemplos de texto
   ```
   (separado por TAB)

3. **Ejecuta el fine-tuning**:
   ```bash
   python scripts/finetune_user_data.py
   ```

## Consejos para mejores resultados

1. **Imágenes de líneas**: El modelo funciona mejor con líneas individuales de texto, no páginas completas.

2. **Contraste**: Asegúrate de que el texto tenga buen contraste con el fondo.

3. **Resolución**: Imágenes de al menos 64px de altura funcionan bien.

4. **Cantidad**: Para fine-tuning, más de 100 muestras darán mejores resultados.

5. **Variedad**: Incluye diferentes estilos de escritura si es posible.

## Formato de annotations.txt

Cada línea debe tener:
```
nombre_imagen.extension<TAB>texto exacto de la imagen
```

Ejemplo:
```
nota_001.png	Comprar leche y pan
nota_002.png	Reunión a las 3pm
receta_01.jpg	2 tazas de harina
```

## Idiomas

El modelo base está entrenado principalmente en inglés, pero puedes hacer fine-tuning para español u otros idiomas con suficientes datos.
