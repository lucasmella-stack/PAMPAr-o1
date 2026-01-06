"""
demo_active_learning.py - Demostración de Active Learning

Script interactivo que demuestra el sistema de active learning completo:
1. Genera datos sintéticos simulando pool no etiquetado
2. Entrena modelo inicial con pequeño dataset seed
3. Ejecuta loop de active learning con diferentes estrategias
4. Visualiza resultados y compara estrategias

Modos de demostración:
- quick: Demo rápido con datos pequeños (5 min)
- full: Demo completo con más iteraciones (30 min)
- compare: Comparar múltiples estrategias
- interactive: Modo interactivo con etiquetado manual simulado

Uso:
    # Demo rápido
    python scripts/demo_active_learning.py --mode quick
    
    # Comparar estrategias
    python scripts/demo_active_learning.py --mode compare --strategies entropy margin diversity
    
    # Demo completo
    python scripts/demo_active_learning.py --mode full --max_iterations 10
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Agregar src al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import torch
import torch.nn as nn


def generate_synthetic_data(
    n_samples: int = 500,
    output_dir: str = "data/synthetic_active_learning",
    image_size: tuple = (224, 224),
    num_classes: int = 3
):
    """
    Genera datos sintéticos para demostración de active learning.
    
    Simula 3 estilos de escritura con características visuales distintas:
    - Clase 0: Líneas horizontales (escritura fluida)
    - Clase 1: Líneas onduladas (escritura temblorosa)
    - Clase 2: Líneas irregulares (escritura difícil)
    
    Args:
        n_samples: Número de muestras a generar
        output_dir: Directorio de salida
        image_size: Tamaño de imágenes
        num_classes: Número de clases
    
    Returns:
        data_df: DataFrame con metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    print(f"🎨 Generando {n_samples} imágenes sintéticas...")
    
    data = []
    
    for i in range(n_samples):
        # Asignar clase
        class_id = i % num_classes
        
        # Crear imagen
        img = Image.new('RGB', image_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Dibujar patrón según clase
        if class_id == 0:
            # Clase 0: Líneas horizontales (fluida)
            for y in range(50, image_size[1] - 50, 30):
                noise = np.random.randint(-5, 5)
                draw.line([(20, y + noise), (image_size[0] - 20, y + noise)], 
                         fill='black', width=2)
        
        elif class_id == 1:
            # Clase 1: Líneas onduladas (temblorosa)
            for y in range(50, image_size[1] - 50, 30):
                points = []
                for x in range(20, image_size[0] - 20, 10):
                    noise = np.random.randint(-10, 10)
                    points.append((x, y + noise))
                draw.line(points, fill='black', width=2)
        
        elif class_id == 2:
            # Clase 2: Líneas irregulares (difícil)
            for y in range(50, image_size[1] - 50, 30):
                for x in range(20, image_size[0] - 20, 5):
                    noise_x = np.random.randint(-3, 3)
                    noise_y = np.random.randint(-15, 15)
                    draw.ellipse(
                        [(x + noise_x - 1, y + noise_y - 1), 
                         (x + noise_x + 1, y + noise_y + 1)],
                        fill='black'
                    )
        
        # Guardar imagen
        image_filename = f"sample_{i:04d}.png"
        image_path = images_dir / image_filename
        img.save(image_path)
        
        # Metadata
        data.append({
            'id': f'sample_{i:04d}',
            'image_path': str(image_path),
            'style_label': class_id,
            'class_name': ['fluida', 'temblorosa', 'irregular'][class_id]
        })
    
    data_df = pd.DataFrame(data)
    
    print(f"✅ {len(data_df)} imágenes generadas en {images_dir}")
    print(f"   Distribución de clases:")
    for class_id in range(num_classes):
        count = (data_df['style_label'] == class_id).sum()
        print(f"     Clase {class_id}: {count} muestras")
    
    return data_df


def split_data(
    data_df: pd.DataFrame,
    seed_size: int = 30,
    val_size: int = 50,
    output_dir: str = "data/synthetic_active_learning"
):
    """
    Divide datos en seed, pool y validación.
    
    Args:
        data_df: DataFrame completo
        seed_size: Tamaño del dataset seed inicial
        val_size: Tamaño del dataset de validación
        output_dir: Directorio de salida
    
    Returns:
        seed_df, pool_df, val_df
    """
    output_path = Path(output_dir)
    
    # Shuffle
    data_shuffled = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split
    val_df = data_shuffled[:val_size]
    seed_df = data_shuffled[val_size:val_size + seed_size]
    pool_df = data_shuffled[val_size + seed_size:]
    
    # Guardar
    seed_path = output_path / "seed.jsonl"
    pool_path = output_path / "pool.jsonl"
    val_path = output_path / "val.jsonl"
    
    seed_df.to_json(seed_path, orient='records', lines=True)
    pool_df.to_json(pool_path, orient='records', lines=True)
    val_df.to_json(val_path, orient='records', lines=True)
    
    print(f"\n📊 Datos divididos:")
    print(f"   Seed: {len(seed_df)} muestras → {seed_path}")
    print(f"   Pool: {len(pool_df)} muestras → {pool_path}")
    print(f"   Val:  {len(val_df)} muestras → {val_path}")
    
    return seed_df, pool_df, val_df


def train_simple_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_classes: int = 3,
    num_epochs: int = 10
):
    """
    Entrena un clasificador simple para demostración.
    
    Usa un CNN pequeño para clasificar las imágenes sintéticas.
    """
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms as T
    
    print("\n🔄 Entrenando clasificador simple...")
    
    # Dataset simple
    class SimpleDataset(Dataset):
        def __init__(self, df, transform=None):
            self.df = df
            self.transform = transform or T.Compose([
                T.Resize((64, 64)),
                T.ToTensor()
            ])
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(row['image_path']).convert('RGB')
            img_tensor = self.transform(img)
            label = int(row['style_label'])
            return img_tensor, label
    
    # DataLoaders
    train_dataset = SimpleDataset(train_df)
    val_dataset = SimpleDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Modelo simple
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(64, num_classes)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validación
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        
        if (epoch + 1) % 3 == 0:
            print(f"   Epoch {epoch+1}/{num_epochs} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {accuracy:.4f}")
    
    print(f"✅ Clasificador entrenado - Accuracy final: {accuracy:.4f}")
    
    return model


def run_active_learning_demo(
    strategy: str = 'entropy',
    n_iterations: int = 5,
    samples_per_iteration: int = 20,
    output_dir: str = "outputs/al_demo"
):
    """
    Ejecuta demo de active learning.
    """
    print("="*70)
    print(f"🚀 DEMO ACTIVE LEARNING - Estrategia: {strategy}")
    print("="*70)
    
    # 1. Generar datos sintéticos
    print("\n📦 Paso 1: Generando datos sintéticos...")
    data_df = generate_synthetic_data(n_samples=300)
    
    # 2. Split datos
    print("\n📊 Paso 2: Dividiendo datos...")
    seed_df, pool_df, val_df = split_data(data_df, seed_size=30, val_size=50)
    
    # 3. Entrenar modelo inicial
    print("\n🎓 Paso 3: Entrenando modelo inicial con seed...")
    model = train_simple_classifier(seed_df, val_df, num_epochs=10)
    
    # 4. Simular active learning loop
    print(f"\n🔄 Paso 4: Ejecutando Active Learning Loop ({n_iterations} iteraciones)...")
    
    from llarri.active_learning.sampler_uncertain import UncertaintySampler
    
    sampler = UncertaintySampler(model, strategy=strategy)
    
    history = {
        'iteration': [0],
        'train_size': [len(seed_df)],
        'val_acc': []
    }
    
    # Evaluación inicial
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms as T
    
    class SimpleDataset(Dataset):
        def __init__(self, df):
            self.df = df
            self.transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(row['image_path']).convert('RGB')
            return self.transform(img), int(row['style_label'])
    
    val_dataset = SimpleDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    initial_acc = correct / total
    history['val_acc'].append(initial_acc)
    
    print(f"   Accuracy inicial: {initial_acc:.4f}")
    
    # Loop de active learning
    current_train = seed_df.copy()
    current_pool = pool_df.copy()
    
    for iteration in range(n_iterations):
        print(f"\n   Iteración {iteration+1}/{n_iterations}")
        
        # Seleccionar muestras
        pool_images = torch.stack([
            T.Compose([T.Resize((64, 64)), T.ToTensor()])(
                Image.open(path).convert('RGB')
            )
            for path in current_pool['image_path']
        ])
        
        selected_indices = sampler.select_samples(
            pool_images, 
            n_samples=min(samples_per_iteration, len(current_pool))
        )
        
        # "Etiquetar" (ya tienen etiquetas en modo oracle)
        new_samples = current_pool.iloc[selected_indices]
        
        # Actualizar datasets
        current_train = pd.concat([current_train, new_samples], ignore_index=True)
        current_pool = current_pool.drop(current_pool.index[selected_indices]).reset_index(drop=True)
        
        # Re-entrenar
        print(f"      Re-entrenando con {len(current_train)} muestras...")
        model = train_simple_classifier(current_train, val_df, num_epochs=5)
        sampler.model = model
        
        # Evaluar
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        
        # Guardar historial
        history['iteration'].append(iteration + 1)
        history['train_size'].append(len(current_train))
        history['val_acc'].append(accuracy)
        
        print(f"      Accuracy: {accuracy:.4f} (Δ={accuracy - history['val_acc'][-2]:+.4f})")
    
    # 5. Visualizar resultados
    print("\n📊 Paso 5: Visualizando resultados...")
    plot_results(history, strategy, output_dir)
    
    print("\n✅ Demo completado")
    print(f"   Accuracy inicial: {history['val_acc'][0]:.4f}")
    print(f"   Accuracy final: {history['val_acc'][-1]:.4f}")
    print(f"   Mejora: {history['val_acc'][-1] - history['val_acc'][0]:+.4f}")
    
    return history


def plot_results(history: dict, strategy: str, output_dir: str):
    """Grafica resultados del active learning loop."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Training size vs accuracy
    ax1.plot(history['train_size'], history['val_acc'], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Training Set Size', fontsize=12)
    ax1.set_ylabel('Validation Accuracy', fontsize=12)
    ax1.set_title(f'Active Learning Curve - {strategy.capitalize()}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Iteration vs accuracy
    ax2.plot(history['iteration'], history['val_acc'], 's-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.set_title('Performance Over Iterations', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    plot_path = output_path / f'al_results_{strategy}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Gráfica guardada en: {plot_path}")


def compare_strategies(strategies: list, n_iterations: int = 5):
    """Compara múltiples estrategias de active learning."""
    print("="*70)
    print("🔬 COMPARACIÓN DE ESTRATEGIAS DE ACTIVE LEARNING")
    print("="*70)
    
    all_histories = {}
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Probando estrategia: {strategy.upper()}")
        print(f"{'='*70}")
        
        history = run_active_learning_demo(
            strategy=strategy,
            n_iterations=n_iterations,
            samples_per_iteration=20
        )
        
        all_histories[strategy] = history
    
    # Comparar resultados
    print("\n" + "="*70)
    print("📊 COMPARACIÓN DE RESULTADOS")
    print("="*70)
    
    plt.figure(figsize=(10, 6))
    
    for strategy, history in all_histories.items():
        plt.plot(
            history['train_size'],
            history['val_acc'],
            'o-',
            label=strategy.capitalize(),
            linewidth=2,
            markersize=6
        )
    
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('Active Learning Strategy Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.tight_layout()
    comparison_path = 'outputs/al_demo/strategy_comparison.png'
    Path('outputs/al_demo').mkdir(parents=True, exist_ok=True)
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Gráfica de comparación guardada en: {comparison_path}")
    
    # Resumen
    print("\n📈 RESUMEN:")
    for strategy, history in all_histories.items():
        improvement = history['val_acc'][-1] - history['val_acc'][0]
        final_acc = history['val_acc'][-1]
        print(f"   {strategy.capitalize():15s} - Final: {final_acc:.4f} - Mejora: {improvement:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Demo de Active Learning")
    
    parser.add_argument(
        '--mode',
        type=str,
        default='quick',
        choices=['quick', 'full', 'compare', 'interactive'],
        help='Modo de demostración'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        default='entropy',
        choices=['least_confidence', 'margin', 'entropy', 'ratio', 'diversity'],
        help='Estrategia de muestreo'
    )
    
    parser.add_argument(
        '--strategies',
        nargs='+',
        default=['entropy', 'margin', 'diversity'],
        help='Estrategias a comparar (modo compare)'
    )
    
    parser.add_argument(
        '--max_iterations',
        type=int,
        default=5,
        help='Número de iteraciones'
    )
    
    parser.add_argument(
        '--samples_per_iteration',
        type=int,
        default=20,
        help='Muestras por iteración'
    )
    
    args = parser.parse_args()
    
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*18 + "ACTIVE LEARNING DEMO" + " "*30 + "║")
    print("╚" + "═"*68 + "╝")
    print("\n")
    
    if args.mode == 'quick':
        print("⚡ Modo rápido - Demo en 5 minutos\n")
        run_active_learning_demo(
            strategy=args.strategy,
            n_iterations=3,
            samples_per_iteration=15
        )
    
    elif args.mode == 'full':
        print("🔬 Modo completo - Demo exhaustivo\n")
        run_active_learning_demo(
            strategy=args.strategy,
            n_iterations=args.max_iterations,
            samples_per_iteration=args.samples_per_iteration
        )
    
    elif args.mode == 'compare':
        print(f"📊 Comparando {len(args.strategies)} estrategias\n")
        compare_strategies(args.strategies, n_iterations=args.max_iterations)
    
    elif args.mode == 'interactive':
        print("🎮 Modo interactivo - Coming soon!\n")
        print("⚠️  Modo interactivo aún no implementado")
    
    print("\n✨ ¡Gracias por probar el demo de Active Learning!\n")


if __name__ == '__main__':
    main()
