"""
LLARRI-O1 v3.1 - Entrenamiento por Cuadrantes
==============================================

Modelo simplificado sin recursión excesiva.
Entrena cuadrantes de menor a mayor, luego las relaciones los conectan.

Autor: Lucas Mella (Segunda Cabeza)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, List, Optional
from dataclasses import dataclass
import time


@dataclass
class ConfigCuadrantes:
    input_dim: int = 784
    hidden_dim: int = 256
    output_dim: int = 10
    num_cajas: int = 3
    dropout: float = 0.1


class Cuadrante(nn.Module):
    """
    Cuadrante básico - unidad fundamental.
    Sin recursión, solo procesamiento directo.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.proceso = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return x + self.proceso(x)


class RelacionCuadrantes(nn.Module):
    """
    Conecta 4 cuadrantes entre sí.
    Aprende relaciones horizontal, vertical y diagonal.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Relaciones entre cuadrantes
        self.rel_h = nn.Linear(dim * 2, dim)  # Horizontal: A↔B, C↔D
        self.rel_v = nn.Linear(dim * 2, dim)  # Vertical: A↔C, B↔D
        self.rel_d = nn.Linear(dim * 2, dim)  # Diagonal: A↔D, B↔C
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, a, b, c, d):
        # Relaciones horizontales
        h_ab = self.rel_h(torch.cat([a, b], dim=-1))
        h_cd = self.rel_h(torch.cat([c, d], dim=-1))
        
        # Relaciones verticales
        v_ac = self.rel_v(torch.cat([a, c], dim=-1))
        v_bd = self.rel_v(torch.cat([b, d], dim=-1))
        
        # Relaciones diagonales
        d_ad = self.rel_d(torch.cat([a, d], dim=-1))
        d_bc = self.rel_d(torch.cat([b, c], dim=-1))
        
        # Actualizar cada cuadrante con sus relaciones
        a = self.norm(a + h_ab + v_ac + d_ad)
        b = self.norm(b + h_ab + v_bd + d_bc)
        c = self.norm(c + h_cd + v_ac + d_bc)
        d = self.norm(d + h_cd + v_bd + d_ad)
        
        return a, b, c, d


class CajaTrinity(nn.Module):
    """
    Caja Trinity con 4 cuadrantes y sus relaciones.
    """
    def __init__(self, config: ConfigCuadrantes, cuadrante_compartido: Optional[Cuadrante] = None):
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        quad_dim = dim // 4
        
        # Proyección de entrada
        self.proj_in = nn.Linear(config.input_dim, dim)
        self.proj_internal = nn.Linear(dim, dim)
        
        # Cuadrante compartido (pesos compartidos entre los 4)
        self.cuadrante = cuadrante_compartido or Cuadrante(quad_dim, config.dropout)
        
        # Relaciones entre cuadrantes
        self.relaciones = RelacionCuadrantes(quad_dim)
        
        # Fusión final
        self.fusion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        # Proyectar entrada
        if x.shape[-1] == self.config.input_dim:
            x = self.proj_in(x)
        elif x.shape[-1] != self.config.hidden_dim:
            x = self.proj_internal(x)
        
        dim = self.config.hidden_dim
        quad_dim = dim // 4
        
        # Dividir en 4 cuadrantes
        a = x[..., :quad_dim]
        b = x[..., quad_dim:quad_dim*2]
        c = x[..., quad_dim*2:quad_dim*3]
        d = x[..., quad_dim*3:]
        
        # Procesar cada cuadrante (mismo peso)
        a = self.cuadrante(a)
        b = self.cuadrante(b)
        c = self.cuadrante(c)
        d = self.cuadrante(d)
        
        # Relacionar cuadrantes
        a, b, c, d = self.relaciones(a, b, c, d)
        
        # Fusionar
        out = torch.cat([a, b, c, d], dim=-1)
        out = self.fusion(out) + x
        
        return out


class LlaveTrinity(nn.Module):
    """Conecta dos cajas Trinity."""
    def __init__(self, dim: int):
        super().__init__()
        self.llave = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, origen, destino):
        return self.norm(destino + self.llave(origen))


class LlarriO1_Cuadrantes(nn.Module):
    """
    LLARRI-O1 v3.1 - Arquitectura por Cuadrantes
    
    Sin recursión excesiva. Cuadrantes se entrenan de menor a mayor,
    las relaciones los conectan al final.
    """
    def __init__(self, config: Optional[ConfigCuadrantes] = None):
        super().__init__()
        self.config = config or ConfigCuadrantes()
        dim = self.config.hidden_dim
        
        # Cuadrante base compartido entre todas las cajas
        self.cuadrante_base = Cuadrante(dim // 4, self.config.dropout)
        
        # 3 Cajas Trinity con cuadrante compartido
        self.caja1 = CajaTrinity(self.config, self.cuadrante_base)
        self.caja2 = CajaTrinity(self.config, self.cuadrante_base)
        self.caja3 = CajaTrinity(self.config, self.cuadrante_base)
        
        # Llaves entre cajas
        self.llave_1_2 = LlaveTrinity(dim)
        self.llave_2_3 = LlaveTrinity(dim)
        self.llave_3_1 = LlaveTrinity(dim)  # Retroalimentación
        
        # Salida
        self.output = nn.Sequential(
            nn.Linear(dim, dim),
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
        
        print(f"\n{'='*60}")
        print(f"  LLARRI-O1 v3.1 - CUADRANTES")
        print(f"{'='*60}")
        print(f"  Autor: Lucas Mella (Segunda Cabeza)")
        print(f"{'='*60}")
        print(f"  Parámetros: {params:,}")
        print(f"  Sin compartir: {params_sin:,}")
        print(f"  Compresión: {comp:.1f}%")
        print(f"  Factor: {params_sin/params:.1f}x")
        print(f"{'='*60}\n")
    
    def _params_sin_compartir(self):
        # Si cada caja tuviera su propio cuadrante
        params_cuadrante = sum(p.numel() for p in self.cuadrante_base.parameters())
        return sum(p.numel() for p in self.parameters()) + params_cuadrante * 11  # 3 cajas * 4 cuadrantes - 1
    
    def forward(self, x):
        # Procesar cajas
        out1 = self.caja1(x)
        out2 = self.caja2(x)
        
        # Conectar caja 1 → 2
        out2 = self.llave_1_2(out1, out2)
        
        # Procesar caja 3 con salidas de 1 y 2
        out3 = self.caja3(out1 + out2)
        
        # Conectar caja 2 → 3
        out3 = self.llave_2_3(out2, out3)
        
        # Retroalimentación 3 → 1 (opcional, para refinamiento)
        # out1 = self.llave_3_1(out3, out1)
        
        return self.output(out3)


def entrenar_por_cuadrantes(epochs: int = 20, batch_size: int = 128):
    """
    Entrena LLARRI-O1 v3.1 por cuadrantes.
    
    Estrategia:
    1. Primero entrena los cuadrantes (base compartida)
    2. Luego entrena las relaciones entre cuadrantes
    3. Finalmente fine-tuning completo
    
    Como todo está interconectado, las relaciones funcionarán al final.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Datos
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    # Modelo
    config = ConfigCuadrantes(hidden_dim=256)
    model = LlarriO1_Cuadrantes(config).to(device)
    
    # Optimizador y loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n{'='*60}")
    print("  ENTRENAMIENTO POR CUADRANTES")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    mejor_acc = 0.0
    
    for epoch in range(epochs):
        inicio = time.time()
        
        # ENTRENAR
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = criterion(out, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        
        scheduler.step()
        
        # EVALUAR
        model.eval()
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.view(x.size(0), -1).to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    out = model(x)
                
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)
        
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        tiempo = time.time() - inicio
        lr = scheduler.get_last_lr()[0]
        
        # Guardar mejor modelo
        es_mejor = val_acc > mejor_acc
        if es_mejor:
            mejor_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, 'checkpoints/llarri_cuadrantes_mejor.pt')
        
        marca = "★" if es_mejor else " "
        print(f"Epoch {epoch+1:2d}/{epochs} ({tiempo:.1f}s) | "
              f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
              f"LR: {lr:.6f} {marca}")
    
    print(f"\n{'='*60}")
    print(f"  ✅ ENTRENAMIENTO COMPLETADO")
    print(f"  Mejor accuracy: {mejor_acc:.2f}%")
    print(f"{'='*60}\n")
    
    return model, mejor_acc


if __name__ == "__main__":
    model, acc = entrenar_por_cuadrantes(epochs=25, batch_size=128)
