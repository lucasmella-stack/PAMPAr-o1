#!/usr/bin/env python3
"""
download_datasets.py - Descarga datasets públicos de manuscritos

Datasets disponibles:
1. IAM Handwriting Database (inglés, pero útil para pretraining)
2. RIMES (francés)
3. CVL (alemán/inglés)
4. Rodrigo Dataset (español histórico) ⭐
5. ESPOSALLES (español histórico) ⭐
6. Washington/Parzival (histórico)
7. Bentham Papers (inglés histórico)
8. ICDAR competitions datasets

Para español específicamente:
- Rodrigo: Manuscritos españoles del siglo XV
- ESPOSALLES: Registros matrimoniales catalán/español 1451-1905
- CARABELA: Manuscritos coloniales

Uso:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --dataset rodrigo
    python scripts/download_datasets.py --list
"""

import os
import sys
import json
import shutil
import hashlib
import logging
import argparse
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Información de un dataset."""
    name: str
    description: str
    language: str
    url: str
    size_mb: int
    license: str
    requires_registration: bool = False
    registration_url: str = ""
    citation: str = ""
    format: str = "images+txt"  # images+txt, xml, json


# Catálogo de datasets disponibles
DATASETS = {
    # =========================================
    # DATASETS EN ESPAÑOL (PRIORIDAD)
    # =========================================
    "rodrigo": DatasetInfo(
        name="Rodrigo Dataset",
        description="Manuscritos españoles del siglo XV. 853 páginas del 'Crónica de España'",
        language="español",
        url="https://zenodo.org/record/1490008/files/rodrigo.tar.gz",
        size_mb=450,
        license="CC BY-NC-SA 4.0",
        requires_registration=False,
        citation="Sánchez, J.A., Romero, V., Toselli, A.H., & Vidal, E. (2016)"
    ),
    
    "esposalles": DatasetInfo(
        name="ESPOSALLES Database",
        description="Registros matrimoniales catalán/español 1451-1905. ~180 páginas",
        language="español/catalán",
        url="https://zenodo.org/record/1322666/files/esposalles.tar.gz",
        size_mb=200,
        license="CC BY-NC-SA 4.0",
        requires_registration=False,
        citation="Romero, V., Fornés, A., et al."
    ),
    
    "carabela": DatasetInfo(
        name="CARABELA",
        description="Manuscritos coloniales latinoamericanos siglos XVI-XVIII",
        language="español",
        url="",  # Requiere solicitud
        size_mb=500,
        license="Research only",
        requires_registration=True,
        registration_url="https://sites.google.com/view/carabela-dataset",
        citation="CARABELA Project"
    ),
    
    # Dataset sintético que podemos generar
    "spanish_synthetic": DatasetInfo(
        name="Spanish Synthetic HTR",
        description="Dataset sintético generado con fuentes manuscritas españolas",
        language="español",
        url="GENERATE",  # Lo generamos nosotros
        size_mb=100,
        license="Free",
        requires_registration=False,
    ),
    
    # =========================================
    # DATASETS MULTILINGÜES (útiles para transfer)
    # =========================================
    "iam": DatasetInfo(
        name="IAM Handwriting Database",
        description="El dataset estándar para HTR. 1539 páginas, 13353 líneas",
        language="inglés",
        url="",  # Requiere registro
        size_mb=800,
        license="Research only",
        requires_registration=True,
        registration_url="https://fki.tic.heia-fr.ch/databases/iam-handwriting-database",
        citation="Marti, U.V., & Bunke, H. (2002)"
    ),
    
    "rimes": DatasetInfo(
        name="RIMES Database",
        description="Cartas manuscritas en francés. ~12000 páginas",
        language="francés",
        url="",  # Requiere registro
        size_mb=1500,
        license="Research only",
        requires_registration=True,
        registration_url="http://www.a2ialab.com/doku.php?id=rimes_database:start",
        citation="Grosicki, E., & El Abed, H. (2009)"
    ),
    
    "bentham": DatasetInfo(
        name="Bentham Papers",
        description="Manuscritos históricos de Jeremy Bentham",
        language="inglés",
        url="https://zenodo.org/record/44519/files/BenthamDatasetR0-GT.zip",
        size_mb=300,
        license="CC BY 4.0",
        requires_registration=False,
        citation="Transkribus Bentham Project"
    ),
    
    "washington": DatasetInfo(
        name="George Washington Papers",
        description="Cartas de George Washington, manuscritos históricos",
        language="inglés",
        url="",  # Múltiples fuentes
        size_mb=150,
        license="Public domain",
        requires_registration=False,
    ),
    
    "cvl": DatasetInfo(
        name="CVL Database",
        description="Computer Vision Lab database, múltiples escritores",
        language="alemán/inglés",
        url="https://zenodo.org/record/1492267/files/cvl-database-1-1.zip",
        size_mb=600,
        license="CC BY-NC-SA 4.0",
        requires_registration=False,
    ),
    
    # =========================================
    # DATASETS DE COMPETICIONES
    # =========================================
    "icdar2017_htr": DatasetInfo(
        name="ICDAR 2017 HTR Competition",
        description="Dataset de competición con múltiples estilos",
        language="varios",
        url="",
        size_mb=400,
        license="Competition",
        requires_registration=True,
        registration_url="https://scriptnet.iit.demokritos.gr/competitions/",
    ),
}


class DownloadProgress:
    """Progress bar para descargas."""
    
    def __init__(self, total_size: int, desc: str = "Downloading"):
        self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=desc)
    
    def update(self, block_num: int, block_size: int, total_size: int):
        self.pbar.update(block_size)
    
    def close(self):
        self.pbar.close()


class DatasetDownloader:
    """Descargador de datasets."""
    
    def __init__(self, output_dir: str = "data/external"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def list_datasets(self) -> None:
        """Lista todos los datasets disponibles."""
        print("\n📚 DATASETS DISPONIBLES PARA HTR\n")
        print("=" * 70)
        
        # Agrupar por idioma
        by_language = {}
        for name, info in DATASETS.items():
            lang = info.language
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append((name, info))
        
        for lang, datasets in by_language.items():
            print(f"\n🌍 {lang.upper()}")
            print("-" * 50)
            
            for name, info in datasets:
                status = "🔒 Requiere registro" if info.requires_registration else "✅ Descarga directa"
                print(f"\n  {name}")
                print(f"    {info.description}")
                print(f"    Tamaño: ~{info.size_mb}MB | {status}")
                if info.requires_registration:
                    print(f"    Registro: {info.registration_url}")
        
        print("\n" + "=" * 70)
        print("\nUso:")
        print("  python scripts/download_datasets.py --dataset rodrigo")
        print("  python scripts/download_datasets.py --dataset rodrigo,bentham,esposalles")
        print("  python scripts/download_datasets.py --spanish  # Solo español")
        print()
    
    def download(self, dataset_name: str) -> Optional[Path]:
        """Descarga un dataset específico."""
        if dataset_name not in DATASETS:
            logger.error(f"Dataset '{dataset_name}' no encontrado")
            return None
        
        info = DATASETS[dataset_name]
        
        if info.requires_registration:
            logger.warning(f"'{dataset_name}' requiere registro manual")
            logger.info(f"Registrate en: {info.registration_url}")
            logger.info(f"Luego coloca los archivos en: {self.output_dir / dataset_name}")
            return None
        
        if info.url == "GENERATE":
            logger.info(f"'{dataset_name}' se genera sintéticamente")
            return self._generate_synthetic(dataset_name)
        
        if not info.url:
            logger.warning(f"'{dataset_name}' no tiene URL de descarga directa")
            return None
        
        # Crear directorio
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Descargar
        logger.info(f"Descargando {info.name} ({info.size_mb}MB)...")
        
        try:
            # Determinar nombre del archivo
            filename = info.url.split("/")[-1]
            filepath = dataset_dir / filename
            
            if filepath.exists():
                logger.info(f"Archivo ya existe: {filepath}")
            else:
                # Descargar con progress bar
                progress = DownloadProgress(info.size_mb * 1024 * 1024, desc=dataset_name)
                urllib.request.urlretrieve(info.url, filepath, reporthook=progress.update)
                progress.close()
            
            # Extraer
            extracted_dir = self._extract(filepath, dataset_dir)
            
            logger.info(f"✅ {dataset_name} descargado en: {extracted_dir}")
            return extracted_dir
            
        except Exception as e:
            logger.error(f"Error descargando {dataset_name}: {e}")
            return None
    
    def _extract(self, filepath: Path, output_dir: Path) -> Path:
        """Extrae archivos comprimidos."""
        logger.info(f"Extrayendo {filepath.name}...")
        
        if filepath.suffix == '.zip':
            with zipfile.ZipFile(filepath, 'r') as zf:
                zf.extractall(output_dir)
        elif filepath.suffix == '.gz' and filepath.stem.endswith('.tar'):
            with tarfile.open(filepath, 'r:gz') as tf:
                tf.extractall(output_dir)
        elif filepath.suffix == '.tar':
            with tarfile.open(filepath, 'r') as tf:
                tf.extractall(output_dir)
        
        return output_dir
    
    def _generate_synthetic(self, dataset_name: str) -> Path:
        """Genera dataset sintético en español."""
        from PIL import Image, ImageDraw, ImageFont
        import random
        
        logger.info("Generando dataset sintético en español...")
        
        output_dir = self.output_dir / dataset_name
        output_dir.mkdir(exist_ok=True)
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Textos de ejemplo en español
        spanish_texts = [
            "Buenos días, ¿cómo está usted?",
            "El rápido zorro marrón salta",
            "La casa de mi abuela",
            "Querido amigo, te escribo",
            "El sol brilla sobre el campo",
            "Las montañas nevadas",
            "Un día de primavera",
            "Los niños juegan en el parque",
            "La música es el alma del pueblo",
            "El tiempo vuela cuando uno",
            "Estimado señor director",
            "Por la presente me dirijo",
            "Atentamente le saluda",
            "Con el debido respeto",
            "Me permito informarle que",
            "Agradezco su atención",
            "Quedo a su disposición",
            "Sin otro particular",
            "Esperando su respuesta",
            "Reciba un cordial saludo",
            # Nombres comunes
            "María García López",
            "Juan Rodríguez Martínez",
            "Carlos Hernández Ruiz",
            "Ana Fernández Sánchez",
            # Direcciones
            "Calle Mayor número 25",
            "Avenida de la Constitución",
            "Plaza del Ayuntamiento",
            # Fechas
            "15 de marzo de 1985",
            "22 de noviembre de 2023",
            "1 de enero del año nuevo",
            # Números
            "Teléfono: 555-123-4567",
            "Documento: 12345678-A",
        ]
        
        # Generar samples
        labels = []
        
        # Intentar usar fuentes del sistema
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            None  # Default
        ]
        
        font = None
        for fp in font_paths:
            if fp and Path(fp).exists():
                try:
                    font = ImageFont.truetype(fp, 32)
                    break
                except:
                    continue
        
        if font is None:
            font = ImageFont.load_default()
        
        for i, text in enumerate(tqdm(spanish_texts * 10, desc="Generando")):  # 10x cada texto
            # Crear imagen
            img_width = max(400, len(text) * 20)
            img_height = 64
            
            # Variaciones de fondo
            bg_color = random.randint(230, 255)
            img = Image.new('RGB', (img_width, img_height), color=(bg_color, bg_color, bg_color))
            draw = ImageDraw.Draw(img)
            
            # Color de texto variable
            text_color = random.randint(0, 50)
            
            # Posición con variación
            x_offset = random.randint(5, 20)
            y_offset = random.randint(10, 20)
            
            draw.text((x_offset, y_offset), text, font=font, fill=(text_color, text_color, text_color))
            
            # Guardar
            img_name = f"sample_{i:05d}.png"
            img.save(images_dir / img_name)
            
            labels.append({
                "image": img_name,
                "text": text,
            })
        
        # Guardar labels
        with open(output_dir / "labels.json", 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        
        # También en formato línea por línea
        with open(output_dir / "labels.txt", 'w', encoding='utf-8') as f:
            for item in labels:
                f.write(f"{item['image']}\t{item['text']}\n")
        
        logger.info(f"✅ Generados {len(labels)} samples sintéticos")
        return output_dir
    
    def download_spanish(self) -> Dict[str, Path]:
        """Descarga todos los datasets en español disponibles."""
        spanish_datasets = [
            name for name, info in DATASETS.items() 
            if "español" in info.language.lower() or "spanish" in info.language.lower()
        ]
        
        results = {}
        for name in spanish_datasets:
            path = self.download(name)
            if path:
                results[name] = path
        
        return results
    
    def download_all_free(self) -> Dict[str, Path]:
        """Descarga todos los datasets que no requieren registro."""
        free_datasets = [
            name for name, info in DATASETS.items()
            if not info.requires_registration and info.url
        ]
        
        results = {}
        for name in free_datasets:
            path = self.download(name)
            if path:
                results[name] = path
        
        return results


class DatasetPreparer:
    """Prepara datasets en formato unificado para LLARRI."""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare(self, dataset_name: str, source_dir: Path) -> Path:
        """Prepara un dataset en formato unificado."""
        
        if dataset_name == "rodrigo":
            return self._prepare_rodrigo(source_dir)
        elif dataset_name == "esposalles":
            return self._prepare_esposalles(source_dir)
        elif dataset_name == "bentham":
            return self._prepare_bentham(source_dir)
        elif dataset_name == "spanish_synthetic":
            return self._prepare_synthetic(source_dir)
        else:
            logger.warning(f"Preparador no implementado para {dataset_name}")
            return source_dir
    
    def _prepare_rodrigo(self, source_dir: Path) -> Path:
        """Prepara dataset Rodrigo."""
        output = self.output_dir / "rodrigo"
        output.mkdir(exist_ok=True)
        
        # Rodrigo tiene estructura: page_XXX.png + page_XXX.txt
        images_dir = output / "images"
        images_dir.mkdir(exist_ok=True)
        
        labels = []
        
        # Buscar archivos
        for img_file in source_dir.rglob("*.png"):
            txt_file = img_file.with_suffix(".txt")
            if txt_file.exists():
                # Copiar imagen
                shutil.copy(img_file, images_dir / img_file.name)
                
                # Leer texto
                with open(txt_file, encoding='utf-8') as f:
                    text = f.read().strip()
                
                labels.append({
                    "image": img_file.name,
                    "text": text,
                })
        
        # Guardar labels unificados
        with open(output / "labels.json", 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Rodrigo: {len(labels)} samples preparados")
        return output
    
    def _prepare_esposalles(self, source_dir: Path) -> Path:
        """Prepara dataset ESPOSALLES."""
        output = self.output_dir / "esposalles"
        output.mkdir(exist_ok=True)
        
        # Similar a Rodrigo
        return self._prepare_generic(source_dir, output)
    
    def _prepare_bentham(self, source_dir: Path) -> Path:
        """Prepara dataset Bentham."""
        output = self.output_dir / "bentham"
        output.mkdir(exist_ok=True)
        
        return self._prepare_generic(source_dir, output)
    
    def _prepare_synthetic(self, source_dir: Path) -> Path:
        """Prepara dataset sintético (ya está en formato correcto)."""
        output = self.output_dir / "spanish_synthetic"
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(source_dir, output)
        return output
    
    def _prepare_generic(self, source_dir: Path, output: Path) -> Path:
        """Preparador genérico para datasets imagen+txt."""
        images_dir = output / "images"
        images_dir.mkdir(exist_ok=True)
        
        labels = []
        
        # Buscar pares imagen-texto
        for img_ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            for img_file in source_dir.rglob(img_ext):
                # Buscar archivo de texto correspondiente
                for txt_ext in ['.txt', '.gt.txt', '.transcription']:
                    txt_file = img_file.with_suffix(txt_ext)
                    if txt_file.exists():
                        # Copiar imagen
                        dest_name = f"{img_file.stem}.png"
                        shutil.copy(img_file, images_dir / dest_name)
                        
                        # Leer texto
                        with open(txt_file, encoding='utf-8', errors='ignore') as f:
                            text = f.read().strip()
                        
                        labels.append({
                            "image": dest_name,
                            "text": text,
                        })
                        break
        
        # Guardar
        with open(output / "labels.json", 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Preparados {len(labels)} samples en {output}")
        return output
    
    def create_splits(self, dataset_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, Path]:
        """Crea splits train/val/test."""
        import random
        
        # Cargar labels
        labels_file = dataset_dir / "labels.json"
        if not labels_file.exists():
            logger.error(f"No existe {labels_file}")
            return {}
        
        with open(labels_file) as f:
            labels = json.load(f)
        
        # Shuffle
        random.seed(42)
        random.shuffle(labels)
        
        # Split
        n = len(labels)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        splits = {
            "train": labels[:train_end],
            "val": labels[train_end:val_end],
            "test": labels[val_end:],
        }
        
        # Guardar splits
        splits_dir = dataset_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        paths = {}
        for split_name, split_data in splits.items():
            split_file = splits_dir / f"{split_name}.json"
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            paths[split_name] = split_file
            logger.info(f"  {split_name}: {len(split_data)} samples")
        
        return paths


def main():
    parser = argparse.ArgumentParser(description="Descargar datasets para HTR")
    parser.add_argument("--list", action="store_true", help="Listar datasets disponibles")
    parser.add_argument("--dataset", type=str, help="Dataset(s) a descargar (separados por coma)")
    parser.add_argument("--spanish", action="store_true", help="Descargar todos los datasets en español")
    parser.add_argument("--all", action="store_true", help="Descargar todos los datasets sin registro")
    parser.add_argument("--output", type=str, default="data/external", help="Directorio de salida")
    parser.add_argument("--prepare", action="store_true", help="Preparar datasets después de descargar")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.output)
    preparer = DatasetPreparer()
    
    if args.list:
        downloader.list_datasets()
        return
    
    downloaded = {}
    
    if args.dataset:
        for name in args.dataset.split(","):
            name = name.strip()
            path = downloader.download(name)
            if path:
                downloaded[name] = path
    
    elif args.spanish:
        downloaded = downloader.download_spanish()
    
    elif args.all:
        downloaded = downloader.download_all_free()
    
    else:
        # Por defecto, mostrar ayuda
        downloader.list_datasets()
        return
    
    # Preparar si se solicita
    if args.prepare and downloaded:
        print("\n📦 Preparando datasets...\n")
        for name, path in downloaded.items():
            prepared = preparer.prepare(name, path)
            preparer.create_splits(prepared)
    
    # Resumen
    if downloaded:
        print("\n" + "=" * 50)
        print("✅ DESCARGA COMPLETADA")
        print("=" * 50)
        for name, path in downloaded.items():
            print(f"  {name}: {path}")
        
        print("\nPróximo paso:")
        print("  python scripts/download_datasets.py --dataset <name> --prepare")
        print("  # O directamente iniciar el frontend de entrenamiento:")
        print("  python scripts/training_server.py")


if __name__ == "__main__":
    main()
