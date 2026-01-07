"""
LLARRI-O1 v3.0 - Entrenamiento Progresivo por Cuadrantes
=========================================================

INNOVACIÓN: Entrenar nivel por nivel, desde el más pequeño
al más grande, congelando pesos anteriores.

Beneficios:
- Reduce RAM dramáticamente en cada paso
- Entrenamiento más estable
- Los niveles pequeños aprenden patrones básicos primero
- Los niveles grandes aprenden a COMBINAR esos patrones

Concepto: "Curriculum Learning Fractal"

Autor: Lucas Mella (Segunda Cabeza)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Optional, Dict, Tuple, List
import time
from pathlib import Path
import json
import math

from llarri_o1_fractal_profundo import (
    LlarriO1_FractalProfundo, 
    LlarriFractalConfig,
    CuadranteFractal,
    CajaTrinityFractal,
    crear_modelo_fractal
)


# ==============================================================================
# MODELO ENTRENABLE POR NIVELES
# ==============================================================================

class LlarriO1_EntrenableProgresivo(nn.Module):
    """
    LLARRI-O1 con entrenamiento progresivo por niveles fractales.
    
    Permite entrenar cada nivel de profundidad por separado,
    congelando los niveles ya entrenados.
    
    Flujo de entrenamiento:
    1. Crear modelo con profundidad deseada
    2. Entrenar nivel N (más profundo) → congelar
    3. Entrenar nivel N-1 → congelar
    4. ... hasta nivel 0
    5. Fine-tuning final (opcional)
    """
    
    def __init__(self, config: Optional[LlarriFractalConfig] = None):
        super().__init__()
        self.config = config or LlarriFractalConfig()
        
        # Detectar dispositivo
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        self.profundidad = self.config.get_profundidad_efectiva()
        self.dims = self.config.get_dims_por_nivel()
        
        # Estado de entrenamiento por nivel
        self.niveles_entrenados = [False] * (self.profundidad + 1)
        self.nivel_actual = self.profundidad  # Empezar desde el más profundo
        
        # Crear capas por nivel (de profundo a superficial)
        self._crear_capas_por_nivel()
        
        # Proyecciones de entrada/salida
        self.input_proj = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        self.output_proj = nn.Linear(self.config.hidden_dim, self.config.output_dim)
        
        self.to(self.device)
        self._print_info()
    
    def _crear_capas_por_nivel(self):
        """Crea las capas para cada nivel fractal"""
        self.capas_nivel = nn.ModuleDict()
        
        for nivel in range(self.profundidad + 1):
            dim = self.dims[nivel]
            
            # Capa de procesamiento para este nivel
            self.capas_nivel[f'proceso_{nivel}'] = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(dim, dim)
            )
            
            # Relaciones posicionales (si no es el nivel más profundo)
            if nivel < self.profundidad:
                sub_dim = self.dims[nivel + 1]
                self.capas_nivel[f'rel_h_{nivel}'] = nn.Linear(sub_dim * 2, sub_dim)
                self.capas_nivel[f'rel_v_{nivel}'] = nn.Linear(sub_dim * 2, sub_dim)
                self.capas_nivel[f'rel_d_{nivel}'] = nn.Linear(sub_dim * 2, sub_dim)
                self.capas_nivel[f'fusion_{nivel}'] = nn.Linear(sub_dim * 4, dim)
        
        # Llaves entre cajas (nivel 0)
        dim = self.dims[0]
        self.llave_1_2 = nn.Linear(dim, dim)
        self.llave_2_1 = nn.Linear(dim, dim)
        self.llave_1_3 = nn.Linear(dim, dim)
        self.llave_2_3 = nn.Linear(dim, dim)
        self.llave_3_1 = nn.Linear(dim, dim)
        
        # Fusión de cajas
        self.fusion_cajas = nn.Linear(dim * 3, dim)
    
    def congelar_nivel(self, nivel: int):
        """Congela los pesos de un nivel específico"""
        for name, param in self.capas_nivel.named_parameters():
            if f'_{nivel}' in name:
                param.requires_grad = False
        
        self.niveles_entrenados[nivel] = True
        print(f"  ❄️  Nivel {nivel} congelado ({self.dims[nivel]} dims)")
    
    def descongelar_nivel(self, nivel: int):
        """Descongela los pesos de un nivel"""
        for name, param in self.capas_nivel.named_parameters():
            if f'_{nivel}' in name:
                param.requires_grad = True
        print(f"  🔥 Nivel {nivel} descongelado ({self.dims[nivel]} dims)")
    
    def congelar_todos(self):
        """Congela todos los niveles"""
        for nivel in range(self.profundidad + 1):
            self.congelar_nivel(nivel)
    
    def descongelar_todos(self):
        """Descongela todos los niveles (para fine-tuning)"""
        for param in self.parameters():
            param.requires_grad = True
        print("  🔥 Todos los niveles descongelados")
    
    def get_parametros_nivel(self, nivel: int) -> int:
        """Retorna el número de parámetros de un nivel"""
        total = 0
        for name, param in self.capas_nivel.named_parameters():
            if f'_{nivel}' in name:
                total += param.numel()
        return total
    
    def get_ram_estimada_nivel(self, nivel: int, batch_size: int = 32) -> float:
        """Estima la RAM necesaria para entrenar un nivel (en MB)"""
        params = self.get_parametros_nivel(nivel)
        dim = self.dims[nivel]
        
        # Parámetros (float32 = 4 bytes)
        ram_params = params * 4 / 1e6
        
        # Gradientes (igual que parámetros)
        ram_grads = ram_params
        
        # Activaciones (aproximado)
        ram_activations = batch_size * dim * 4 * 10 / 1e6  # factor 10 por capas intermedias
        
        # Optimizador (Adam usa 2x parámetros)
        ram_optimizer = ram_params * 2
        
        return ram_params + ram_grads + ram_activations + ram_optimizer
    
    def procesar_nivel(self, x: torch.Tensor, nivel: int) -> torch.Tensor:
        """Procesa un tensor a través de un nivel específico"""
        dim = self.dims[nivel]
        
        # Ajustar dimensión si es necesario
        if x.shape[-1] != dim:
            if x.shape[-1] > dim:
                x = x[..., :dim]
            else:
                x = torch.nn.functional.pad(x, (0, dim - x.shape[-1]))
        
        # Procesamiento interno del nivel
        x_procesado = x + self.capas_nivel[f'proceso_{nivel}'](x)
        
        # Si no es el nivel más profundo, procesar sub-cuadrantes
        if nivel < self.profundidad:
            sub_dim = self.dims[nivel + 1]
            
            # Dividir en 4 sub-cuadrantes
            s1 = x_procesado[..., :sub_dim]
            s2 = x_procesado[..., sub_dim:sub_dim*2]
            s3 = x_procesado[..., sub_dim*2:sub_dim*3]
            s4 = x_procesado[..., sub_dim*3:sub_dim*4]
            
            # Procesar recursivamente cada sub-cuadrante
            s1 = self.procesar_nivel(s1, nivel + 1)
            s2 = self.procesar_nivel(s2, nivel + 1)
            s3 = self.procesar_nivel(s3, nivel + 1)
            s4 = self.procesar_nivel(s4, nivel + 1)
            
            # Relaciones posicionales
            rel_h = self.capas_nivel[f'rel_h_{nivel}'](torch.cat([s1, s2], dim=-1))
            rel_v = self.capas_nivel[f'rel_v_{nivel}'](torch.cat([s1, s3], dim=-1))
            rel_d = self.capas_nivel[f'rel_d_{nivel}'](torch.cat([s1, s4], dim=-1))
            
            # Actualizar sub-cuadrantes
            s1 = s1 + rel_h + rel_v + rel_d
            s2 = s2 + rel_h
            s3 = s3 + rel_v
            s4 = s4 + rel_d
            
            # Fusionar
            fusion = torch.cat([s1, s2, s3, s4], dim=-1)
            x_procesado = self.capas_nivel[f'fusion_{nivel}'](fusion)
        
        return x_procesado
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass completo"""
        x = x.to(self.device)
        
        # Proyectar entrada
        x = self.input_proj(x)
        
        # Procesar 3 cajas (mismo procesamiento, diferentes "caminos")
        caja1 = self.procesar_nivel(x, 0)
        caja2 = self.procesar_nivel(x, 0)
        caja3 = self.procesar_nivel(x, 0)
        
        # Llaves entre cajas
        caja1 = caja1 + self.llave_2_1(caja2) + self.llave_3_1(caja3)
        caja2 = caja2 + self.llave_1_2(caja1)
        caja3 = caja3 + self.llave_1_3(caja1) + self.llave_2_3(caja2)
        
        # Fusionar cajas
        fusion = torch.cat([caja1, caja2, caja3], dim=-1)
        output = self.fusion_cajas(fusion)
        
        # Proyectar salida
        output = self.output_proj(output)
        
        return output
    
    def _print_info(self):
        """Muestra información del modelo"""
        print(f"\n{'='*70}")
        print(f"  LLARRI-O1 v3.0 - ENTRENAMIENTO PROGRESIVO")
        print(f"{'='*70}")
        print(f"  Autor: Lucas Mella (Segunda Cabeza)")
        print(f"{'='*70}")
        print(f"\n  NIVELES FRACTALES:")
        
        total_params = 0
        for nivel in range(self.profundidad + 1):
            params = self.get_parametros_nivel(nivel)
            ram = self.get_ram_estimada_nivel(nivel)
            total_params += params
            estado = "❄️" if self.niveles_entrenados[nivel] else "🔥"
            print(f"  {estado} Nivel {nivel}: {self.dims[nivel]:4d} dims | "
                  f"{params:,} params | ~{ram:.1f} MB RAM")
        
        print(f"\n  Total parámetros: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  Dispositivo: {self.device}")
        print(f"{'='*70}\n")


# ==============================================================================
# ENTRENADOR PROGRESIVO
# ==============================================================================

class EntrenadorProgresivo:
    """
    Entrenador que entrena nivel por nivel.
    
    Estrategia:
    1. Empezar por el nivel más profundo (menos RAM)
    2. Entrenar hasta convergencia
    3. Congelar y pasar al siguiente nivel
    4. Repetir hasta nivel 0
    5. Fine-tuning opcional (todo descongelado)
    """
    
    def __init__(
        self,
        modelo: LlarriO1_EntrenableProgresivo,
        lr_base: float = 1e-3
    ):
        self.modelo = modelo
        self.device = modelo.device
        self.lr_base = lr_base
        
        self.historial_por_nivel = {}
        
    def entrenar_nivel(
        self,
        nivel: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr: Optional[float] = None
    ) -> Dict:
        """
        Entrena un nivel específico.
        
        Los niveles más profundos usan menos datos porque
        procesan patrones más simples.
        """
        lr = lr or self.lr_base
        
        print(f"\n{'='*60}")
        print(f"  ENTRENANDO NIVEL {nivel} ({self.modelo.dims[nivel]} dims)")
        print(f"{'='*60}")
        print(f"  Learning rate: {lr}")
        print(f"  RAM estimada: {self.modelo.get_ram_estimada_nivel(nivel):.1f} MB")
        
        # Solo optimizar parámetros de este nivel
        params_nivel = [p for name, p in self.modelo.named_parameters() 
                       if f'_{nivel}' in name and p.requires_grad]
        
        # Añadir proyecciones si es nivel 0
        if nivel == 0:
            params_nivel.extend([
                p for p in self.modelo.input_proj.parameters()
            ])
            params_nivel.extend([
                p for p in self.modelo.output_proj.parameters()
            ])
            params_nivel.extend([
                self.modelo.llave_1_2.weight, self.modelo.llave_1_2.bias,
                self.modelo.llave_2_1.weight, self.modelo.llave_2_1.bias,
                self.modelo.llave_1_3.weight, self.modelo.llave_1_3.bias,
                self.modelo.llave_2_3.weight, self.modelo.llave_2_3.bias,
                self.modelo.llave_3_1.weight, self.modelo.llave_3_1.bias,
                self.modelo.fusion_cajas.weight, self.modelo.fusion_cajas.bias
            ])
        
        if not params_nivel:
            print("  ⚠️  No hay parámetros para entrenar en este nivel")
            return {}
        
        optimizer = optim.AdamW(params_nivel, lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        historial = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        mejor_acc = 0
        
        for epoch in range(epochs):
            # Entrenar
            self.modelo.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            for data, target in train_loader:
                data = data.view(data.size(0), -1).to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                output = self.modelo(data)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params_nivel, 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_correct += output.argmax(1).eq(target).sum().item()
                train_total += target.size(0)
            
            # Evaluar
            self.modelo.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.view(data.size(0), -1).to(self.device)
                    target = target.to(self.device)
                    
                    output = self.modelo(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    val_correct += output.argmax(1).eq(target).sum().item()
                    val_total += target.size(0)
            
            # Métricas
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            historial['train_loss'].append(train_loss)
            historial['train_acc'].append(train_acc)
            historial['val_loss'].append(val_loss)
            historial['val_acc'].append(val_acc)
            
            mejor = " ★" if val_acc > mejor_acc else ""
            if val_acc > mejor_acc:
                mejor_acc = val_acc
            
            print(f"  Época {epoch+1:2d}/{epochs} | "
                  f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%{mejor}")
        
        print(f"\n  ✓ Nivel {nivel} completado. Mejor acc: {mejor_acc:.1f}%")
        
        self.historial_por_nivel[nivel] = historial
        return historial
    
    def entrenar_progresivo(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs_por_nivel: int = 10,
        fine_tuning_epochs: int = 5
    ) -> Dict:
        """
        Entrenamiento progresivo completo.
        
        1. Entrena desde el nivel más profundo
        2. Congela cada nivel al terminar
        3. Fine-tuning final opcional
        """
        print(f"\n{'='*70}")
        print(f"  ENTRENAMIENTO PROGRESIVO FRACTAL")
        print(f"  Niveles: {self.modelo.profundidad + 1}")
        print(f"  Épocas por nivel: {epochs_por_nivel}")
        print(f"{'='*70}")
        
        # Entrenar desde el nivel más profundo al 0
        for nivel in range(self.modelo.profundidad, -1, -1):
            self.entrenar_nivel(
                nivel,
                train_loader,
                val_loader,
                epochs=epochs_por_nivel
            )
            
            # Congelar este nivel
            if nivel > 0:  # No congelar el nivel 0 antes del fine-tuning
                self.modelo.congelar_nivel(nivel)
        
        # Fine-tuning (todo descongelado)
        if fine_tuning_epochs > 0:
            print(f"\n{'='*60}")
            print(f"  FINE-TUNING GLOBAL")
            print(f"{'='*60}")
            
            self.modelo.descongelar_todos()
            
            optimizer = optim.AdamW(
                self.modelo.parameters(), 
                lr=self.lr_base * 0.1,  # LR más bajo para fine-tuning
                weight_decay=1e-4
            )
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(fine_tuning_epochs):
                self.modelo.train()
                train_correct, train_total = 0, 0
                
                for data, target in train_loader:
                    data = data.view(data.size(0), -1).to(self.device)
                    target = target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = self.modelo(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_correct += output.argmax(1).eq(target).sum().item()
                    train_total += target.size(0)
                
                # Evaluar
                self.modelo.eval()
                val_correct, val_total = 0, 0
                with torch.no_grad():
                    for data, target in val_loader:
                        data = data.view(data.size(0), -1).to(self.device)
                        target = target.to(self.device)
                        output = self.modelo(data)
                        val_correct += output.argmax(1).eq(target).sum().item()
                        val_total += target.size(0)
                
                train_acc = 100 * train_correct / train_total
                val_acc = 100 * val_correct / val_total
                print(f"  Fine-tune {epoch+1}/{fine_tuning_epochs} | "
                      f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")
        
        print(f"\n{'='*70}")
        print(f"  ✓ ENTRENAMIENTO PROGRESIVO COMPLETADO")
        print(f"{'='*70}\n")
        
        return self.historial_por_nivel


# ==============================================================================
# FUNCIONES DE UTILIDAD
# ==============================================================================

def cargar_mnist_reducido(
    batch_size: int = 128, 
    porcentaje: float = 1.0,
    data_dir: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    """
    Carga MNIST con opción de reducir el dataset.
    
    Útil para entrenar niveles profundos con menos datos.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # Reducir si es necesario
    if porcentaje < 1.0:
        n_train = int(len(train_dataset) * porcentaje)
        n_val = int(len(val_dataset) * porcentaje)
        
        train_indices = torch.randperm(len(train_dataset))[:n_train]
        val_indices = torch.randperm(len(val_dataset))[:n_val]
        
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def entrenar_modelo_progresivo(
    hidden_dim: int = 256,
    epochs_por_nivel: int = 10,
    fine_tuning_epochs: int = 5,
    batch_size: int = 128
) -> Tuple[LlarriO1_EntrenableProgresivo, Dict]:
    """
    Función principal para entrenamiento progresivo.
    """
    print("\n" + "="*70)
    print("  LLARRI-O1 v3.0 - ENTRENAMIENTO PROGRESIVO POR CUADRANTES")
    print("="*70)
    print("  Autor: Lucas Mella (Segunda Cabeza)")
    print("="*70)
    
    # Cargar datos
    print("\n  Cargando MNIST...")
    train_loader, val_loader = cargar_mnist_reducido(batch_size=batch_size)
    print(f"  ✓ Train: {len(train_loader.dataset):,} imágenes")
    print(f"  ✓ Val: {len(val_loader.dataset):,} imágenes")
    
    # Crear modelo
    config = LlarriFractalConfig(
        input_dim=784,
        hidden_dim=hidden_dim,
        output_dim=10
    )
    modelo = LlarriO1_EntrenableProgresivo(config)
    
    # Crear entrenador
    entrenador = EntrenadorProgresivo(modelo)
    
    # Entrenar
    historial = entrenador.entrenar_progresivo(
        train_loader,
        val_loader,
        epochs_por_nivel=epochs_por_nivel,
        fine_tuning_epochs=fine_tuning_epochs
    )
    
    return modelo, historial


# ==============================================================================
# DEMO: COMPARAR RAM POR NIVEL
# ==============================================================================

def demo_comparar_ram():
    """Muestra el ahorro de RAM al entrenar por niveles"""
    print("\n" + "="*70)
    print("  COMPARACIÓN DE RAM POR NIVEL")
    print("="*70)
    
    config = LlarriFractalConfig(hidden_dim=256)
    modelo = LlarriO1_EntrenableProgresivo(config)
    
    print("\n  Si entrenaras TODO junto:")
    ram_total = sum(modelo.get_ram_estimada_nivel(n) for n in range(modelo.profundidad + 1))
    print(f"  RAM necesaria: ~{ram_total:.1f} MB")
    
    print("\n  Entrenando por NIVELES:")
    ram_max = max(modelo.get_ram_estimada_nivel(n) for n in range(modelo.profundidad + 1))
    print(f"  RAM máxima necesaria: ~{ram_max:.1f} MB")
    
    print(f"\n  🎯 AHORRO: {(1 - ram_max/ram_total)*100:.1f}% menos RAM!")
    print("="*70 + "\n")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # Mostrar comparación de RAM
    demo_comparar_ram()
    
    # Entrenar modelo
    modelo, historial = entrenar_modelo_progresivo(
        hidden_dim=256,
        epochs_por_nivel=5,  # Reducido para demo
        fine_tuning_epochs=3,
        batch_size=128
    )
    
    # Guardar
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': modelo.state_dict(),
        'config': modelo.config,
        'historial': historial
    }, "checkpoints/modelo_progresivo.pt")
    
    print("  ✓ Modelo guardado en checkpoints/modelo_progresivo.pt")
