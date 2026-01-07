"""
LLARRI-O1 v4.0 - HyperComprimido (Entrenamiento Progresivo)
============================================================

Entrenamiento SECUENCIAL desde el cuadrante más pequeño:
2 → 4 → 8 → 16 → 32 → 64

NO procesa todo en paralelo - va nivel por nivel, cuadrante por cuadrante.

Autor: Lucas Mella (Segunda Cabeza)
Coordinador: Alvaro (Segunda Cabeza)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import os


@dataclass
class ConfigHyperComprimido:
    """Configuración del modelo HyperComprimido."""
    input_dim: int = 784
    hidden_dim: int = 256
    output_dim: int = 10
    num_cajas_datos: int = 3
    num_cajas_calculos: int = 3
    # 6 niveles: del más pequeño al más grande (optimizado)
    niveles_fractales: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64])
    dropout: float = 0.1


class CacheBinario:
    """
    Cache en RAM para el nivel binario (dim=2).
    Pre-computa TODAS las operaciones posibles - lookup instantáneo.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self._tabla = None
        self._inicializar()
    
    def _inicializar(self):
        """Pre-computa tabla de lookup para todas las combinaciones."""
        # 4 estados posibles × 7 operaciones
        self._tabla = torch.zeros(4, 7, device=self.device)
        
        combinaciones = [
            [0.0, 0.0],  # Estado 0
            [0.0, 1.0],  # Estado 1
            [1.0, 0.0],  # Estado 2
            [1.0, 1.0],  # Estado 3
        ]
        
        for i, (a, b) in enumerate(combinaciones):
            self._tabla[i] = torch.tensor([
                a + b,                      # suma
                a * b,                      # producto
                abs(a - b),                 # diferencia
                (a + b) / 2,                # media
                max(a, b),                  # máximo
                min(a, b),                  # mínimo
                a * (1-b) + (1-a) * b,      # XOR suave
            ], device=self.device)
    
    def to(self, device):
        self.device = device
        self._inicializar()
        return self
    
    def lookup(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lookup vectorizado en la tabla.
        x: tensor de shape (batch, 2)
        returns: tensor de shape (batch, 7)
        """
        # Cuantizar a índice 0-3: [a,b] -> a*2 + b
        x_bin = (x > 0.5).float()
        idx = (x_bin[..., 0] * 2 + x_bin[..., 1]).long()
        return self._tabla[idx]


class ProcesoNivel(nn.Module):
    """Procesa UN nivel específico."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.proceso = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.proceso(x))


class CuadranteProgresivo(nn.Module):
    """
    Cuadrante que procesa SECUENCIALMENTE desde nivel 2 hasta 64.
    
    Flujo: entrada(64) → comprimir(2) → 2 → 4 → 8 → 16 → 32 → 64 → salida
    """
    def __init__(self, dim_cuadrante: int, config: ConfigHyperComprimido, 
                 cache_binario: Optional[CacheBinario] = None):
        super().__init__()
        self.dim = dim_cuadrante
        self.niveles = config.niveles_fractales  # [2, 4, 8, 16, 32, 64]
        self.cache = cache_binario
        
        # Comprimir entrada al nivel binario
        self.comprimir = nn.Linear(dim_cuadrante, 2)
        
        # Fusión de cache binario
        self.fusion_cache = nn.Linear(2 + 7, 2)
        
        # Procesos para cada nivel
        self.procesos = nn.ModuleDict()
        for nivel in self.niveles:
            self.procesos[str(nivel)] = ProcesoNivel(nivel, config.dropout)
        
        # Transiciones entre niveles (subir)
        self.subir = nn.ModuleDict()
        for i in range(len(self.niveles) - 1):
            din, dout = self.niveles[i], self.niveles[i + 1]
            self.subir[f'{din}_{dout}'] = nn.Linear(din, dout)
        
        # Expandir al tamaño original
        self.expandir = nn.Linear(self.niveles[-1], dim_cuadrante)
        self.norm_final = nn.LayerNorm(dim_cuadrante)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Procesa secuencialmente: 2 → 4 → 8 → 16 → 32 → 64"""
        residual = x
        
        # 1. Comprimir al nivel binario
        h = self.comprimir(x)  # (batch, 2)
        
        # 2. Usar cache binario (lookup instantáneo)
        if self.cache is not None:
            cache_out = self.cache.lookup(h)  # (batch, 7)
            h = self.fusion_cache(torch.cat([h, cache_out], dim=-1))
        
        # 3. Procesar nivel 2
        h = self.procesos['2'](h)
        
        # 4. Subir SECUENCIALMENTE por cada nivel
        for i in range(len(self.niveles) - 1):
            din, dout = self.niveles[i], self.niveles[i + 1]
            h = self.subir[f'{din}_{dout}'](h)
            h = self.procesos[str(dout)](h)
        
        # 5. Expandir y residual
        h = self.expandir(h)
        return self.norm_final(h + residual)


class RelacionesCuadrantes(nn.Module):
    """Conecta 4 cuadrantes entre sí."""
    def __init__(self, dim: int):
        super().__init__()
        self.rel = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, a, b, c, d):
        # Relaciones cruzadas
        ab = self.rel(torch.cat([a, b], dim=-1))
        cd = self.rel(torch.cat([c, d], dim=-1))
        ac = self.rel(torch.cat([a, c], dim=-1))
        bd = self.rel(torch.cat([b, d], dim=-1))
        
        a = self.norm(a + ab + ac)
        b = self.norm(b + ab + bd)
        c = self.norm(c + cd + ac)
        d = self.norm(d + cd + bd)
        
        return a, b, c, d


class CajaDatos(nn.Module):
    """Caja de DATOS - procesa 4 cuadrantes secuencialmente."""
    def __init__(self, config: ConfigHyperComprimido, cuadrante: CuadranteProgresivo):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        quad_dim = dim // 4
        
        self.proj_in = nn.Linear(config.input_dim, dim)
        self.cuadrante = cuadrante
        self.relaciones = RelacionesCuadrantes(quad_dim)
        self.fusion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.config.input_dim:
            x = self.proj_in(x)
        
        quad_dim = self.config.hidden_dim // 4
        
        # Dividir en 4 cuadrantes
        a = x[..., :quad_dim]
        b = x[..., quad_dim:quad_dim*2]
        c = x[..., quad_dim*2:quad_dim*3]
        d = x[..., quad_dim*3:]
        
        # Procesar SECUENCIALMENTE cada cuadrante
        a = self.cuadrante(a)
        b = self.cuadrante(b)
        c = self.cuadrante(c)
        d = self.cuadrante(d)
        
        # Relacionar
        a, b, c, d = self.relaciones(a, b, c, d)
        
        out = torch.cat([a, b, c, d], dim=-1)
        return self.fusion(out) + x


class CajaCalculos(nn.Module):
    """Caja de CÁLCULOS - opera sobre datos + otros cálculos."""
    def __init__(self, config: ConfigHyperComprimido, cuadrante: CuadranteProgresivo):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        quad_dim = dim // 4
        
        self.op_combinar = nn.Linear(dim * 2, dim)
        self.cuadrante = cuadrante
        self.relaciones = RelacionesCuadrantes(quad_dim)
        self.integrar = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, datos1: torch.Tensor, datos2: torch.Tensor,
                otros_calc: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.op_combinar(torch.cat([datos1, datos2], dim=-1))
        
        if otros_calc is not None:
            x = self.norm(x + self.integrar(torch.cat([x, otros_calc], dim=-1)) * 0.5)
        
        quad_dim = self.config.hidden_dim // 4
        
        a = x[..., :quad_dim]
        b = x[..., quad_dim:quad_dim*2]
        c = x[..., quad_dim*2:quad_dim*3]
        d = x[..., quad_dim*3:]
        
        a = self.cuadrante(a)
        b = self.cuadrante(b)
        c = self.cuadrante(c)
        d = self.cuadrante(d)
        
        a, b, c, d = self.relaciones(a, b, c, d)
        
        return torch.cat([a, b, c, d], dim=-1) + x


class LlaveConexion(nn.Module):
    """Conecta cajas entre sí."""
    def __init__(self, dim: int):
        super().__init__()
        self.llave = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, origen: torch.Tensor, destino: torch.Tensor) -> torch.Tensor:
        return self.norm(destino + self.llave(origen) * 0.5)


class LlarriO1_HyperComprimido(nn.Module):
    """
    LLARRI-O1 v4.0 - HyperComprimido
    
    Entrenamiento PROGRESIVO SECUENCIAL:
    - Nivel binario (2) primero
    - Subir: 2 → 4 → 8 → 16 → 32 → 64
    - Cuadrante por cuadrante
    - Caja por caja
    
    NO procesa todo en paralelo - evita lentitud.
    
    Autor: Lucas Mella (Segunda Cabeza)
    """
    
    def __init__(self, config: Optional[ConfigHyperComprimido] = None):
        super().__init__()
        self.config = config or ConfigHyperComprimido()
        dim = self.config.hidden_dim
        quad_dim = dim // 4
        
        # Cache binario
        self.cache_binario = CacheBinario(torch.device('cpu'))
        
        # UN cuadrante compartido para TODO el modelo
        self.cuadrante_base = CuadranteProgresivo(
            quad_dim, self.config, self.cache_binario
        )
        
        # === CAPA DE DATOS (3 cajas) ===
        self.caja_datos_A = CajaDatos(self.config, self.cuadrante_base)
        self.caja_datos_B = CajaDatos(self.config, self.cuadrante_base)
        self.caja_datos_C = CajaDatos(self.config, self.cuadrante_base)
        
        # === CAPA DE CÁLCULOS (3 cajas) ===
        self.caja_calc_A = CajaCalculos(self.config, self.cuadrante_base)
        self.caja_calc_B = CajaCalculos(self.config, self.cuadrante_base)
        self.caja_calc_C = CajaCalculos(self.config, self.cuadrante_base)
        
        # === LLAVES ===
        self.llave_AB = LlaveConexion(dim)
        self.llave_BC = LlaveConexion(dim)
        self.llave_CA = LlaveConexion(dim)
        
        self.llave_calc_AB = LlaveConexion(dim)
        self.llave_calc_BC = LlaveConexion(dim)
        
        self.llave_datos_calc_A = LlaveConexion(dim)
        self.llave_datos_calc_B = LlaveConexion(dim)
        self.llave_datos_calc_C = LlaveConexion(dim)
        
        # === SALIDA ===
        self.output = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(dim, self.config.output_dim)
        )
        
        self._print_info()
    
    def _print_info(self):
        params = sum(p.numel() for p in self.parameters())
        params_sin = self._params_sin_compartir()
        comp = (1 - params/params_sin) * 100
        
        print(f"\n{'='*70}")
        print(f"{'LLARRI-O1 v4.0 - HYPERCOMPRIMIDO (PROGRESIVO)':^70}")
        print(f"{'='*70}")
        print(f"  Autor: Lucas Mella (Segunda Cabeza)")
        print(f"  Coordinador: Alvaro (Segunda Cabeza)")
        print(f"{'='*70}")
        print(f"  ARQUITECTURA:")
        print(f"    • Cajas de datos: {self.config.num_cajas_datos}")
        print(f"    • Cajas de cálculos: {self.config.num_cajas_calculos}")
        print(f"    • Total cajas: 6")
        print(f"    • Niveles fractales: {len(self.config.niveles_fractales)}")
        print(f"    • Flujo: {' → '.join(map(str, self.config.niveles_fractales))}")
        print(f"    • Cache binario: ✓ Activado")
        print(f"{'='*70}")
        print(f"  ENTRENAMIENTO PROGRESIVO:")
        print(f"    • Empieza en nivel 2 (binario)")
        print(f"    • Sube secuencialmente hasta 64")
        print(f"    • Cuadrante por cuadrante")
        print(f"    • Sin procesamiento paralelo pesado")
        print(f"{'='*70}")
        print(f"  PARÁMETROS:")
        print(f"    • Reales: {params:,}")
        print(f"    • Sin compartir: {params_sin:,}")
        print(f"    • Compresión: {comp:.1f}%")
        print(f"    • Tamaño: {params * 4 / 1e6:.2f} MB")
        print(f"{'='*70}\n")
    
    def _params_sin_compartir(self) -> int:
        params_cuadrante = sum(p.numel() for p in self.cuadrante_base.parameters())
        return sum(p.numel() for p in self.parameters()) + params_cuadrante * 23
    
    def to(self, device):
        super().to(device)
        self.cache_binario.to(device)
        return self
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass SECUENCIAL."""
        # === CAPA DE DATOS ===
        out_A = self.caja_datos_A(x)
        out_B = self.caja_datos_B(x)
        out_B = self.llave_AB(out_A, out_B)
        
        out_C = self.caja_datos_C(x)
        out_C = self.llave_BC(out_B, out_C)
        out_A = self.llave_CA(out_C, out_A)
        
        # === CAPA DE CÁLCULOS ===
        calc_A = self.caja_calc_A(out_A, out_B, None)
        calc_B = self.caja_calc_B(out_B, out_C, calc_A)
        calc_B = self.llave_calc_AB(calc_A, calc_B)
        
        calc_C = self.caja_calc_C(out_C, out_A, calc_B)
        calc_C = self.llave_calc_BC(calc_B, calc_C)
        
        # === RETROALIMENTACIÓN ===
        out_A = self.llave_datos_calc_A(calc_A, out_A)
        out_B = self.llave_datos_calc_B(calc_B, out_B)
        out_C = self.llave_datos_calc_C(calc_C, out_C)
        
        # === FUSIÓN ===
        datos_final = out_A + out_B + out_C
        calc_final = calc_A + calc_B + calc_C
        
        fusion = torch.cat([datos_final, calc_final], dim=-1)
        return self.output(fusion)


def entrenar_progresivo(epochs: int = 25, batch_size: int = 128):
    """Entrena LLARRI-O1 v4.0 con flujo progresivo."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"{'ENTRENAMIENTO PROGRESIVO':^70}")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Flujo: 2 → 4 → 8 → 16 → 32 → 64")
    print(f"{'='*70}\n")
    
    # Datos
    transform_train = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST('data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, 
                             num_workers=2, pin_memory=True)
    
    # Modelo
    config = ConfigHyperComprimido()
    model = LlarriO1_HyperComprimido(config).to(device)
    
    # Optimizador
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    patience = 5
    no_improve = 0
    
    for epoch in range(epochs):
        start = time.time()
        
        # Training
        model.train()
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(data.size(0), -1).to(device)
                target = target.to(device)
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    output = model(data)
                
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        elapsed = time.time() - start
        lr = scheduler.get_last_lr()[0]
        
        scheduler.step()
        
        # Early stopping
        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            no_improve = 0
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
                'config': config,
            }, 'checkpoints/llarri_hypercomprimido_mejor.pt')
            marker = '★'
        else:
            no_improve += 1
            marker = ' '
        
        print(f"Epoch {epoch+1:2d}/{epochs} ({elapsed:.1f}s) | "
              f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
              f"LR: {lr:.6f} {marker}")
        
        if no_improve >= patience:
            print(f"\n⚠️  Early stopping: sin mejora en {patience} epochs")
            break
    
    print(f"\n{'='*70}")
    print(f"  ✅ ENTRENAMIENTO COMPLETADO")
    print(f"  Mejor accuracy: {best_acc:.2f}%")
    print(f"{'='*70}\n")
    
    return model, best_acc


if __name__ == "__main__":
    model, acc = entrenar_progresivo(epochs=25, batch_size=128)
