#!/usr/bin/env python3
"""
generate_large_synthetic.py - Generador de dataset sintético grande y variado

Genera miles de muestras de texto manuscrito sintético en español con:
- Múltiples fuentes
- Variaciones de estilo (negrita, cursiva, etc.)
- Diferentes tamaños
- Augmentaciones (ruido, rotación, blur, etc.)
- Textos variados (nombres, direcciones, fechas, frases, números)

Uso:
    python scripts/generate_large_synthetic.py --samples 5000
    python scripts/generate_large_synthetic.py --samples 10000 --augment
"""

import os
import sys
import json
import random
import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
from tqdm import tqdm

# Configuración base
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "external"


# ============================================
# CORPUS DE TEXTOS EN ESPAÑOL
# ============================================

NOMBRES_PROPIOS = [
    "María", "José", "Carmen", "Antonio", "Ana", "Francisco", "Isabel", "Juan",
    "Lucía", "Manuel", "Rosa", "Carlos", "Pilar", "Miguel", "Teresa", "Luis",
    "Dolores", "Rafael", "Juana", "Fernando", "Esperanza", "Pedro", "Mercedes",
    "Alejandro", "Sofía", "Pablo", "Laura", "Diego", "Paula", "Jorge", "Elena",
    "Martín", "Beatriz", "Andrés", "Clara", "Gonzalo", "Adriana", "Felipe", "Silvia",
    "Ramón", "Alicia", "Roberto", "Mónica", "Ángel", "Verónica", "Sergio", "Patricia",
    "Emilio", "Cristina", "Ricardo", "Sandra", "Víctor", "Marta", "Alberto", "Rocío",
]

APELLIDOS = [
    "García", "Rodríguez", "Martínez", "López", "González", "Hernández", "Pérez",
    "Sánchez", "Ramírez", "Torres", "Flores", "Rivera", "Gómez", "Díaz", "Morales",
    "Reyes", "Cruz", "Ortiz", "Gutiérrez", "Chávez", "Ramos", "Vargas", "Castillo",
    "Mendoza", "Romero", "Herrera", "Medina", "Aguilar", "Vega", "Castro", "Jiménez",
    "Ruiz", "Álvarez", "Muñoz", "Fernández", "Suárez", "Blanco", "Molina", "Delgado",
    "Ortega", "Guerrero", "Santos", "Núñez", "Campos", "Vázquez", "León", "Domínguez",
]

CALLES = [
    "Calle Mayor", "Avenida de la Constitución", "Plaza del Ayuntamiento",
    "Calle Real", "Paseo de la Castellana", "Gran Vía", "Calle del Carmen",
    "Avenida Libertad", "Calle San José", "Paseo del Prado", "Calle Santa María",
    "Avenida de España", "Calle Nueva", "Plaza Mayor", "Calle del Sol",
    "Avenida del Parque", "Calle de la Paz", "Paseo Marítimo", "Calle Victoria",
    "Avenida Central", "Calle Ancha", "Plaza de Armas", "Calle del Norte",
    "Avenida Sur", "Calle Independencia", "Paseo de los Álamos", "Calle Florida",
]

CIUDADES = [
    "Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza", "Málaga", "Murcia",
    "Palma", "Bilbao", "Alicante", "Córdoba", "Valladolid", "Vigo", "Gijón",
    "Granada", "A Coruña", "Vitoria", "Elche", "Oviedo", "Pamplona", "Santander",
    "Burgos", "Salamanca", "Toledo", "Cádiz", "Segovia", "Ávila", "Cuenca",
    "Buenos Aires", "Ciudad de México", "Bogotá", "Lima", "Santiago", "Caracas",
]

PAISES = [
    "España", "México", "Argentina", "Colombia", "Perú", "Venezuela", "Chile",
    "Ecuador", "Guatemala", "Cuba", "Bolivia", "Honduras", "Paraguay", "Nicaragua",
    "El Salvador", "Costa Rica", "Panamá", "Uruguay", "Puerto Rico", "Rep. Dominicana",
]

MESES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
]

FRASES_FORMALES = [
    "Estimado señor director",
    "Por la presente me dirijo a usted",
    "Me permito informarle que",
    "Agradezco su atención",
    "Quedo a su disposición",
    "Sin otro particular",
    "Esperando su pronta respuesta",
    "Reciba un cordial saludo",
    "Con el debido respeto",
    "Atentamente le saluda",
    "A quien corresponda",
    "Muy señor mío",
    "Distinguido colega",
    "Apreciado cliente",
    "Mediante la presente comunicación",
    "Por medio de la presente",
    "Le escribo para informarle",
    "Me complace comunicarle",
    "Lamento informarle que",
    "Es mi deber informarle",
    "Hago constar que",
    "Certifico que el señor",
    "Para los efectos legales",
    "En cumplimiento de lo establecido",
    "De conformidad con la ley",
]

FRASES_COTIDIANAS = [
    "Buenos días, ¿cómo está usted?",
    "Muchas gracias por su ayuda",
    "El rápido zorro marrón salta",
    "La casa de mi abuela está lejos",
    "El sol brilla sobre el campo",
    "Las montañas nevadas del norte",
    "Un hermoso día de primavera",
    "Los niños juegan en el parque",
    "La música es el alma del pueblo",
    "El tiempo vuela cuando te diviertes",
    "No hay mal que por bien no venga",
    "A caballo regalado no le mires el diente",
    "Más vale tarde que nunca",
    "En boca cerrada no entran moscas",
    "Quien mucho abarca poco aprieta",
    "El que madruga Dios le ayuda",
    "Ojos que no ven corazón que no siente",
    "Hoy es un buen día para empezar",
    "La paciencia es una virtud",
    "Todo tiene solución menos la muerte",
    "El trabajo dignifica al hombre",
    "La educación es la base del progreso",
    "Unidos somos más fuertes",
    "El conocimiento es poder",
    "La familia es lo primero",
]

PROFESIONES = [
    "abogado", "médico", "ingeniero", "profesor", "contador", "arquitecto",
    "enfermero", "dentista", "periodista", "farmacéutico", "veterinario",
    "economista", "psicólogo", "sociólogo", "administrador", "secretario",
    "electricista", "plomero", "carpintero", "mecánico", "chofer", "cocinero",
]

DOCUMENTOS = [
    "Documento Nacional de Identidad",
    "Pasaporte español",
    "Licencia de conducir",
    "Certificado de nacimiento",
    "Acta de matrimonio",
    "Título universitario",
    "Contrato de trabajo",
    "Escritura de propiedad",
    "Testamento del señor",
    "Poder notarial",
    "Certificado médico",
    "Constancia de residencia",
]


# ============================================
# GENERADORES DE TEXTO
# ============================================

def generar_nombre_completo() -> str:
    """Genera nombre completo aleatorio."""
    nombre = random.choice(NOMBRES_PROPIOS)
    ap1 = random.choice(APELLIDOS)
    ap2 = random.choice(APELLIDOS)
    return f"{nombre} {ap1} {ap2}"


def generar_direccion() -> str:
    """Genera dirección aleatoria."""
    calle = random.choice(CALLES)
    numero = random.randint(1, 200)
    piso = random.choice(["", f", piso {random.randint(1, 10)}", f", {random.randint(1, 10)}º {random.choice(['A', 'B', 'C', 'D'])}"])
    return f"{calle} {numero}{piso}"


def generar_fecha() -> str:
    """Genera fecha aleatoria."""
    dia = random.randint(1, 28)
    mes = random.choice(MESES)
    año = random.randint(1950, 2025)
    formatos = [
        f"{dia} de {mes} de {año}",
        f"{dia}/{MESES.index(mes)+1:02d}/{año}",
        f"{dia}-{MESES.index(mes)+1:02d}-{año}",
        f"{mes} {dia}, {año}",
    ]
    return random.choice(formatos)


def generar_telefono() -> str:
    """Genera teléfono aleatorio."""
    prefijos = ["91", "93", "94", "95", "96", "98", "922", "928", "971", "986"]
    prefijo = random.choice(prefijos)
    numero = "".join([str(random.randint(0, 9)) for _ in range(7)])
    formatos = [
        f"{prefijo} {numero[:3]} {numero[3:]}",
        f"{prefijo}-{numero[:3]}-{numero[3:]}",
        f"+34 {prefijo} {numero}",
    ]
    return f"Tel: {random.choice(formatos)}"


def generar_dni() -> str:
    """Genera DNI aleatorio."""
    numero = random.randint(10000000, 99999999)
    letra = random.choice("ABCDEFGHJKLMNPQRSTVWXYZ")
    return f"DNI: {numero}-{letra}"


def generar_email() -> str:
    """Genera email aleatorio."""
    nombre = random.choice(NOMBRES_PROPIOS).lower()
    apellido = random.choice(APELLIDOS).lower()
    dominios = ["gmail.com", "hotmail.com", "yahoo.es", "outlook.com", "empresa.es"]
    numero = random.choice(["", str(random.randint(1, 99))])
    return f"{nombre}.{apellido}{numero}@{random.choice(dominios)}"


def generar_cantidad() -> str:
    """Genera cantidad monetaria."""
    cantidad = random.randint(10, 100000)
    formatos = [
        f"{cantidad:,}€".replace(",", "."),
        f"EUR {cantidad:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
        f"{cantidad} euros",
    ]
    return random.choice(formatos)


def generar_texto_aleatorio() -> str:
    """Genera texto aleatorio de diferentes tipos."""
    tipo = random.choice([
        "nombre", "nombre", "nombre",  # Mayor probabilidad
        "direccion", "direccion",
        "fecha", "fecha",
        "telefono", "dni", "email", "cantidad",
        "frase_formal", "frase_formal",
        "frase_cotidiana", "frase_cotidiana",
        "documento", "profesion",
        "ciudad", "combinado"
    ])
    
    if tipo == "nombre":
        return generar_nombre_completo()
    elif tipo == "direccion":
        return generar_direccion()
    elif tipo == "fecha":
        return generar_fecha()
    elif tipo == "telefono":
        return generar_telefono()
    elif tipo == "dni":
        return generar_dni()
    elif tipo == "email":
        return generar_email()
    elif tipo == "cantidad":
        return generar_cantidad()
    elif tipo == "frase_formal":
        return random.choice(FRASES_FORMALES)
    elif tipo == "frase_cotidiana":
        return random.choice(FRASES_COTIDIANAS)
    elif tipo == "documento":
        return random.choice(DOCUMENTOS)
    elif tipo == "profesion":
        nombre = generar_nombre_completo()
        prof = random.choice(PROFESIONES)
        return f"{nombre}, {prof}"
    elif tipo == "ciudad":
        ciudad = random.choice(CIUDADES)
        pais = random.choice(PAISES)
        return f"{ciudad}, {pais}"
    else:  # combinado
        nombre = generar_nombre_completo()
        dir = generar_direccion()
        return f"{nombre} - {dir}"


# ============================================
# GENERADOR DE IMÁGENES
# ============================================

@dataclass
class FontConfig:
    path: str
    sizes: List[int]
    is_handwriting: bool = False


def get_available_fonts() -> List[FontConfig]:
    """Obtiene fuentes disponibles en el sistema."""
    font_candidates = [
        # Fuentes serif (simulan escritura formal)
        ("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", [24, 28, 32, 36]),
        ("/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf", [24, 28, 32, 36]),
        ("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", [24, 28, 32]),
        
        # Fuentes sans (texto impreso)
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", [24, 28, 32, 36]),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf", [24, 28, 32]),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", [24, 28, 32]),
        
        # DejaVu
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", [24, 28, 32, 36]),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", [24, 28, 32, 36]),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", [24, 28, 32]),
        
        # Noto
        ("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", [24, 28, 32, 36]),
        ("/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf", [24, 28, 32, 36]),
        
        # URW fonts (Open Type)
        ("/usr/share/fonts/opentype/urw-base35/P052-Roman.otf", [24, 28, 32]),
        ("/usr/share/fonts/opentype/urw-base35/P052-Italic.otf", [24, 28, 32]),
        ("/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf", [24, 28, 32]),
        ("/usr/share/fonts/opentype/urw-base35/NimbusSans-Italic.otf", [24, 28, 32]),
    ]
    
    available = []
    for path, sizes in font_candidates:
        if Path(path).exists():
            available.append(FontConfig(path, sizes))
    
    return available if available else [FontConfig("", [24, 28, 32])]  # Default


def apply_augmentations(img: Image.Image, intensity: str = "medium") -> Image.Image:
    """Aplica augmentaciones a la imagen."""
    
    if intensity == "none":
        return img
    
    # Convertir a numpy para algunas operaciones
    img_array = np.array(img)
    
    # 1. Ruido gaussiano (30% probabilidad)
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(3, 10), img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
    
    # 2. Variación de brillo (40% probabilidad)
    if random.random() < 0.4:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.85, 1.15))
    
    # 3. Variación de contraste (30% probabilidad)
    if random.random() < 0.3:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.9, 1.1))
    
    # 4. Blur ligero (20% probabilidad)
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    
    # 5. Rotación pequeña (25% probabilidad)
    if intensity == "high" and random.random() < 0.25:
        angle = random.uniform(-2, 2)
        img = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)
    
    # 6. Simulación de tinta corrida (15% probabilidad)
    if random.random() < 0.15:
        img = img.filter(ImageFilter.SMOOTH)
    
    return img


def generate_sample(
    text: str,
    font_config: FontConfig,
    sample_id: int,
    output_dir: Path,
    augment: bool = True,
    augment_intensity: str = "medium"
) -> Optional[dict]:
    """Genera una muestra individual."""
    try:
        # Cargar fuente
        font_size = random.choice(font_config.sizes)
        if font_config.path:
            font = ImageFont.truetype(font_config.path, font_size)
        else:
            font = ImageFont.load_default()
        
        # Calcular dimensiones
        dummy_img = Image.new('RGB', (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Dimensiones de imagen con padding variable
        padding_x = random.randint(15, 40)
        padding_y = random.randint(10, 25)
        img_width = max(200, text_width + 2 * padding_x)
        img_height = max(48, text_height + 2 * padding_y)
        
        # Color de fondo variable (simula papel)
        bg_variations = [
            (255, 255, 255),  # Blanco puro
            (252, 252, 250),  # Blanco cálido
            (250, 248, 245),  # Crema claro
            (248, 245, 240),  # Beige muy claro
            (245, 245, 245),  # Gris muy claro
            (255, 253, 248),  # Amarillento
        ]
        bg_color = random.choice(bg_variations)
        
        # Crear imagen
        img = Image.new('RGB', (img_width, img_height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Color de texto variable (simula diferentes tintas)
        text_colors = [
            (0, 0, 0),        # Negro
            (20, 20, 20),     # Gris muy oscuro
            (30, 30, 40),     # Negro azulado
            (40, 30, 30),     # Negro rojizo
            (0, 0, 50),       # Azul muy oscuro
            (50, 30, 0),      # Marrón oscuro
        ]
        text_color = random.choice(text_colors)
        
        # Posición del texto
        x = padding_x + random.randint(-5, 5)
        y = padding_y + random.randint(-3, 3)
        
        # Dibujar texto
        draw.text((x, y), text, font=font, fill=text_color)
        
        # Aplicar augmentaciones
        if augment:
            img = apply_augmentations(img, augment_intensity)
        
        # Guardar
        img_name = f"syn_{sample_id:06d}.png"
        img_path = output_dir / "images" / img_name
        img.save(img_path, "PNG", optimize=True)
        
        return {
            "image": img_name,
            "text": text,
            "font": Path(font_config.path).name if font_config.path else "default",
            "font_size": font_size,
        }
        
    except Exception as e:
        print(f"Error generando muestra {sample_id}: {e}")
        return None


def generate_dataset(
    num_samples: int,
    output_name: str = "spanish_synthetic_large",
    augment: bool = True,
    augment_intensity: str = "medium",
    num_workers: int = None
) -> Path:
    """Genera el dataset completo."""
    
    output_dir = DATA_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"GENERADOR DE DATASET SINTÉTICO EN ESPAÑOL")
    print(f"{'='*60}")
    print(f"📁 Directorio: {output_dir}")
    print(f"📊 Muestras a generar: {num_samples:,}")
    print(f"🎨 Augmentaciones: {augment} ({augment_intensity})")
    
    # Obtener fuentes
    fonts = get_available_fonts()
    print(f"🔤 Fuentes disponibles: {len(fonts)}")
    for f in fonts[:5]:
        print(f"   - {Path(f.path).name if f.path else 'default'}")
    if len(fonts) > 5:
        print(f"   ... y {len(fonts)-5} más")
    
    # Generar textos
    print(f"\n📝 Generando textos...")
    texts = [generar_texto_aleatorio() for _ in range(num_samples)]
    
    # Asignar fuentes aleatorias a cada muestra
    font_assignments = [random.choice(fonts) for _ in range(num_samples)]
    
    # Generar muestras
    print(f"\n🖼️  Generando imágenes...")
    labels = []
    
    # Usar multiprocessing para acelerar
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Para evitar problemas con PIL y multiprocessing, usamos un loop simple con progress
    for i, (text, font_config) in enumerate(tqdm(
        zip(texts, font_assignments), 
        total=num_samples,
        desc="Generando"
    )):
        result = generate_sample(
            text=text,
            font_config=font_config,
            sample_id=i,
            output_dir=output_dir,
            augment=augment,
            augment_intensity=augment_intensity
        )
        if result:
            labels.append(result)
    
    # Guardar labels en JSON
    labels_json_path = output_dir / "labels.json"
    with open(labels_json_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    
    # Guardar labels en formato TSV
    labels_tsv_path = output_dir / "labels.tsv"
    with open(labels_tsv_path, 'w', encoding='utf-8') as f:
        f.write("image\ttext\n")
        for item in labels:
            f.write(f"{item['image']}\t{item['text']}\n")
    
    # Crear splits (80% train, 10% val, 10% test)
    random.shuffle(labels)
    n_train = int(len(labels) * 0.8)
    n_val = int(len(labels) * 0.1)
    
    train_labels = labels[:n_train]
    val_labels = labels[n_train:n_train + n_val]
    test_labels = labels[n_train + n_val:]
    
    splits = {
        "train": train_labels,
        "val": val_labels,
        "test": test_labels
    }
    
    for split_name, split_data in splits.items():
        split_path = output_dir / f"{split_name}.json"
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
    
    # Estadísticas
    print(f"\n{'='*60}")
    print(f"✅ DATASET GENERADO EXITOSAMENTE")
    print(f"{'='*60}")
    print(f"📊 Total muestras: {len(labels):,}")
    print(f"   - Train: {len(train_labels):,} ({len(train_labels)/len(labels)*100:.1f}%)")
    print(f"   - Val:   {len(val_labels):,} ({len(val_labels)/len(labels)*100:.1f}%)")
    print(f"   - Test:  {len(test_labels):,} ({len(test_labels)/len(labels)*100:.1f}%)")
    
    # Calcular estadísticas de texto
    all_texts = [l["text"] for l in labels]
    avg_len = sum(len(t) for t in all_texts) / len(all_texts)
    max_len = max(len(t) for t in all_texts)
    min_len = min(len(t) for t in all_texts)
    
    print(f"\n📝 Estadísticas de texto:")
    print(f"   - Longitud promedio: {avg_len:.1f} caracteres")
    print(f"   - Longitud máxima: {max_len} caracteres")
    print(f"   - Longitud mínima: {min_len} caracteres")
    
    # Caracteres únicos
    all_chars = set("".join(all_texts))
    print(f"   - Caracteres únicos: {len(all_chars)}")
    
    # Tamaño en disco
    total_size = sum(f.stat().st_size for f in (output_dir / "images").glob("*.png"))
    print(f"\n💾 Tamaño en disco: {total_size / 1024 / 1024:.1f} MB")
    print(f"📁 Ubicación: {output_dir}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Genera dataset sintético en español")
    parser.add_argument("--samples", type=int, default=5000,
                       help="Número de muestras a generar (default: 5000)")
    parser.add_argument("--output", type=str, default="spanish_synthetic_large",
                       help="Nombre del dataset de salida")
    parser.add_argument("--augment", action="store_true", default=True,
                       help="Aplicar augmentaciones (default: True)")
    parser.add_argument("--no-augment", action="store_true",
                       help="No aplicar augmentaciones")
    parser.add_argument("--intensity", type=str, default="medium",
                       choices=["low", "medium", "high"],
                       help="Intensidad de augmentaciones")
    
    args = parser.parse_args()
    
    augment = not args.no_augment
    
    generate_dataset(
        num_samples=args.samples,
        output_name=args.output,
        augment=augment,
        augment_intensity=args.intensity
    )


if __name__ == "__main__":
    main()
