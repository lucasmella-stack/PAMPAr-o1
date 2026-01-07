# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
LLARRI-O1 - Entrenador Progresivo
==================================

Entrena el modelo nivel por nivel para reducir uso de memoria.

Estrategia:
1. Entrenar nivel más profundo (cuadrantes más pequeños)
2. Congelar y entrenar siguiente nivel
3. Repetir hasta nivel 0
4. Fine-tuning final

Autor: Lucas Mella (Segunda Cabeza)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Any
import time
from pathlib import Path
import copy

from ..models.fractal_profundo import LlarriO1_FractalProfundo, LlarriFractalConfig


class EntrenadorProgresivo:
    """
    Entrenador progresivo nivel por nivel.
    
    Ventajas:
    - Menor uso de memoria por nivel
    - Entrenamiento más estable
    - Permite control granular
    
    Uso:
        entrenador = EntrenadorProgresivo(config)
        modelo = entrenador.entrenar_todos_niveles(train_loader, test_loader)
    """
    
    def __init__(
        self,
        config: Optional[LlarriFractalConfig] = None,
        epochs_por_nivel: int = 5,
        epochs_finetune: int = 10,
        lr_inicial: float = 1e-3,
        lr_finetune: float = 1e-4,
        verbose: bool = True,
        checkpoint_dir: str = "checkpoints/progresivo"
    ):
        self.config = config or LlarriFractalConfig()
        self.epochs_por_nivel = epochs_por_nivel
        self.epochs_finetune = epochs_finetune
        self.lr_inicial = lr_inicial
        self.lr_finetune = lr_finetune
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Crear directorio de checkpoints
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Historia
        self.historia: Dict[str, Any] = {
            'niveles': [],
            'finetune': None
        }
    
    def _log(self, msg: str):
        """Log si verbose"""
        if self.verbose:
            print(msg)
    
    def _obtener_parametros_nivel(self, modelo: nn.Module, nivel: int) -> List[nn.Parameter]:
        """Obtiene parámetros de un nivel específico"""
        params = []
        
        def buscar_nivel(modulo, nivel_buscado, nivel_actual=0):
            for nombre, submodulo in modulo.named_children():
                if 'cuadrante' in nombre.lower() or 'subcuadrante' in nombre.lower():
                    if nivel_actual == nivel_buscado:
                        params.extend(submodulo.parameters())
                    else:
                        buscar_nivel(submodulo, nivel_buscado, nivel_actual + 1)
                else:
                    buscar_nivel(submodulo, nivel_buscado, nivel_actual)
        
        buscar_nivel(modelo, nivel)
        return params
    
    def _congelar_nivel(self, modelo: nn.Module, nivel: int):
        """Congela parámetros de un nivel"""
        params = self._obtener_parametros_nivel(modelo, nivel)
        for p in params:
            p.requires_grad = False
    
    def _descongelar_todo(self, modelo: nn.Module):
        """Descongela todos los parámetros"""
        for p in modelo.parameters():
            p.requires_grad = True
    
    def _entrenar_epoca(
        self,
        modelo: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        criterion: nn.Module
    ) -> tuple:
        """Entrena una época"""
        modelo.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            x, y = batch
            if len(x.shape) > 2:
                x = x.view(x.size(0), -1)
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.device.type=='cuda'):
                outputs = modelo(x)
                loss = criterion(outputs, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        return total_loss / len(train_loader), 100. * correct / total
    
    @torch.no_grad()
    def _evaluar(
        self,
        modelo: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module
    ) -> tuple:
        """Evalúa el modelo"""
        modelo.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in test_loader:
            x, y = batch
            if len(x.shape) > 2:
                x = x.view(x.size(0), -1)
            x, y = x.to(self.device), y.to(self.device)
            
            outputs = modelo(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        return total_loss / len(test_loader), 100. * correct / total
    
    def entrenar_nivel(
        self,
        modelo: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        nivel: int,
        epochs: int,
        lr: float
    ) -> Dict[str, List[float]]:
        """
        Entrena un nivel específico.
        
        Args:
            modelo: Modelo a entrenar
            train_loader: DataLoader de entrenamiento
            test_loader: DataLoader de prueba
            nivel: Nivel a entrenar
            epochs: Número de épocas
            lr: Learning rate
        
        Returns:
            Historia del nivel
        """
        self._log(f"\n{'='*50}")
        self._log(f"ENTRENANDO NIVEL {nivel}")
        self._log(f"{'='*50}")
        
        # Obtener parámetros del nivel
        params = self._obtener_parametros_nivel(modelo, nivel)
        if not params:
            # Si no hay parámetros específicos, usar todos los no congelados
            params = [p for p in modelo.parameters() if p.requires_grad]
        
        optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        
        historia = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        mejor_acc = 0.0
        
        for epoch in range(epochs):
            inicio = time.time()
            
            train_loss, train_acc = self._entrenar_epoca(modelo, optimizer, train_loader, criterion)
            val_loss, val_acc = self._evaluar(modelo, test_loader, criterion)
            
            tiempo = time.time() - inicio
            
            historia['train_loss'].append(train_loss)
            historia['train_acc'].append(train_acc)
            historia['val_loss'].append(val_loss)
            historia['val_acc'].append(val_acc)
            
            scheduler.step(val_acc)
            
            if val_acc > mejor_acc:
                mejor_acc = val_acc
                # Guardar checkpoint del nivel
                torch.save(
                    modelo.state_dict(),
                    f"{self.checkpoint_dir}/nivel_{nivel}_mejor.pt"
                )
            
            self._log(f"  Época {epoch+1}/{epochs} ({tiempo:.1f}s)")
            self._log(f"    Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            self._log(f"    Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        self._log(f"\n  Mejor accuracy nivel {nivel}: {mejor_acc:.2f}%")
        
        return historia
    
    def entrenar_todos_niveles(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader
    ) -> LlarriO1_FractalProfundo:
        """
        Entrena todos los niveles progresivamente.
        
        Proceso:
        1. Crear modelo
        2. Entrenar desde nivel más profundo hacia arriba
        3. Congelar niveles entrenados
        4. Fine-tuning final con todo descongelado
        
        Returns:
            Modelo entrenado
        """
        self._log(f"\n{'='*60}")
        self._log("ENTRENAMIENTO PROGRESIVO POR NIVELES")
        self._log(f"{'='*60}")
        
        # Crear modelo
        modelo = LlarriO1_FractalProfundo(self.config)
        modelo.to(self.device)
        
        profundidad = self.config.get_profundidad_efectiva()
        self._log(f"Profundidad fractal: {profundidad} niveles")
        self._log(f"Device: {self.device}")
        self._log(f"Modo: {modelo.rm.modo}")
        
        # Entrenar nivel por nivel (de profundo a superficial)
        for nivel in range(profundidad, -1, -1):
            historia_nivel = self.entrenar_nivel(
                modelo,
                train_loader,
                test_loader,
                nivel,
                self.epochs_por_nivel,
                self.lr_inicial * (0.5 ** (profundidad - nivel))  # LR decrece
            )
            
            self.historia['niveles'].append({
                'nivel': nivel,
                'historia': historia_nivel
            })
            
            # Congelar nivel entrenado
            self._congelar_nivel(modelo, nivel)
        
        # Fine-tuning final
        self._log(f"\n{'='*50}")
        self._log("FINE-TUNING FINAL")
        self._log(f"{'='*50}")
        
        self._descongelar_todo(modelo)
        
        historia_finetune = self.entrenar_nivel(
            modelo,
            train_loader,
            test_loader,
            nivel=-1,  # -1 indica todos los niveles
            epochs=self.epochs_finetune,
            lr=self.lr_finetune
        )
        
        self.historia['finetune'] = historia_finetune
        
        # Guardar modelo final
        torch.save(
            modelo.state_dict(),
            f"{self.checkpoint_dir}/modelo_final.pt"
        )
        
        # Evaluar final
        _, acc_final = self._evaluar(
            modelo, test_loader, nn.CrossEntropyLoss()
        )
        
        self._log(f"\n{'='*60}")
        self._log("ENTRENAMIENTO PROGRESIVO COMPLETADO")
        self._log(f"{'='*60}")
        self._log(f"Accuracy final: {acc_final:.2f}%")
        self._log(f"Modelo guardado en: {self.checkpoint_dir}/modelo_final.pt")
        self._log(f"{'='*60}\n")
        
        return modelo
    
    def get_resumen(self) -> str:
        """Retorna un resumen del entrenamiento"""
        lineas = ["RESUMEN ENTRENAMIENTO PROGRESIVO", "="*40]
        
        for info_nivel in self.historia['niveles']:
            nivel = info_nivel['nivel']
            hist = info_nivel['historia']
            mejor_acc = max(hist['val_acc']) if hist['val_acc'] else 0
            lineas.append(f"Nivel {nivel}: {mejor_acc:.2f}%")
        
        if self.historia['finetune']:
            mejor_acc = max(self.historia['finetune']['val_acc'])
            lineas.append(f"Fine-tune: {mejor_acc:.2f}%")
        
        return "\n".join(lineas)


# ==============================================================================
# DEMO
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("DEMO: Entrenador Progresivo")
    print("="*50)
    
    from ..utils.datos import cargar_mnist_plano
    
    config = LlarriFractalConfig(
        hidden_dim=128,  # Más pequeño para demo
        profundidad_fractal=2
    )
    
    entrenador = EntrenadorProgresivo(
        config,
        epochs_por_nivel=2,
        epochs_finetune=3,
        verbose=True
    )
    
    # Cargar datos
    train_loader, test_loader = cargar_mnist_plano(batch_size=128)
    
    # Entrenar
    modelo = entrenador.entrenar_todos_niveles(train_loader, test_loader)
    
    print(entrenador.get_resumen())
