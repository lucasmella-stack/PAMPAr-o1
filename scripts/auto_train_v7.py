# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
LLARRI v7 - AUTOENTRENAMIENTO ADAPTATIVO

Sistema de meta-learning donde el modelo:
1. Guarda configuraciones que producen buenos resultados
2. Se autoajusta basado en historial de éxitos
3. Aprende qué hiperparámetros funcionan mejor
4. Adapta su comportamiento dinámicamente

Inspirado en:
- Meta-Learning (MAML)
- Bayesian Optimization
- Reinforcement Learning
- Neuroplasticidad cerebral
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import deque
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from llarri_o1.models.language_model_v7 import LLARRIv7Cerebral


class MemoriaAdaptativa:
    """
    Memoria que guarda configuraciones exitosas.
    
    Guarda:
    - Estados del modelo que produjeron buenos resultados
    - Hiperparámetros efectivos
    - Patrones de modulación del tálamo exitosos
    """
    
    def __init__(self, capacidad: int = 1000, path: str = 'checkpoints/memoria_adaptativa.json'):
        self.capacidad = capacidad
        self.path = path
        self.memoria = {
            'configuraciones_exitosas': deque(maxlen=capacidad),
            'historial_lr': [],
            'historial_loss': [],
            'patrones_talamo': deque(maxlen=500),
            'mejores_epochs': [],
            'meta_params': {
                'lr_optimo': 1e-4,
                'temperatura_talamo': 1.0,
                'dropout_optimo': 0.1,
                'momentum_exitoso': 0.0,
            },
            'estadisticas': {
                'total_entrenamientos': 0,
                'mejoras_consecutivas': 0,
                'mejor_loss_historico': float('inf'),
                'peor_loss_historico': 0,
            }
        }
        self._cargar()
    
    def _cargar(self):
        """Carga memoria persistente si existe."""
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                    # Convertir listas a deques
                    self.memoria['configuraciones_exitosas'] = deque(
                        data.get('configuraciones_exitosas', []), 
                        maxlen=self.capacidad
                    )
                    self.memoria['patrones_talamo'] = deque(
                        data.get('patrones_talamo', []),
                        maxlen=500
                    )
                    self.memoria['historial_lr'] = data.get('historial_lr', [])
                    self.memoria['historial_loss'] = data.get('historial_loss', [])
                    self.memoria['meta_params'] = data.get('meta_params', self.memoria['meta_params'])
                    self.memoria['estadisticas'] = data.get('estadisticas', self.memoria['estadisticas'])
                    self.memoria['mejores_epochs'] = data.get('mejores_epochs', [])
                print(f'📚 Memoria adaptativa cargada: {len(self.memoria["configuraciones_exitosas"])} configs')
            except Exception as e:
                print(f'⚠️ Error cargando memoria: {e}')
    
    def guardar(self):
        """Persiste la memoria a disco."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = {
            'configuraciones_exitosas': list(self.memoria['configuraciones_exitosas']),
            'patrones_talamo': list(self.memoria['patrones_talamo']),
            'historial_lr': self.memoria['historial_lr'][-1000:],  # Últimos 1000
            'historial_loss': self.memoria['historial_loss'][-1000:],
            'meta_params': self.memoria['meta_params'],
            'estadisticas': self.memoria['estadisticas'],
            'mejores_epochs': self.memoria['mejores_epochs'][-50:],
        }
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def registrar_exito(self, config: dict, loss: float, modulacion_talamo: dict):
        """Registra una configuración exitosa."""
        self.memoria['configuraciones_exitosas'].append({
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'loss': loss,
            'modulacion': modulacion_talamo,
        })
        
        # Actualizar estadísticas
        stats = self.memoria['estadisticas']
        if loss < stats['mejor_loss_historico']:
            stats['mejor_loss_historico'] = loss
            stats['mejoras_consecutivas'] += 1
        else:
            stats['mejoras_consecutivas'] = 0
        
        if loss > stats['peor_loss_historico']:
            stats['peor_loss_historico'] = loss
        
        stats['total_entrenamientos'] += 1
    
    def registrar_patron_talamo(self, modulacion: dict, loss: float):
        """Guarda patrones de modulación que funcionaron bien."""
        self.memoria['patrones_talamo'].append({
            'modulacion': modulacion,
            'loss': loss,
        })
    
    def obtener_lr_optimo(self) -> float:
        """Calcula el LR óptimo basado en historial."""
        if len(self.memoria['historial_lr']) < 10:
            return self.memoria['meta_params']['lr_optimo']
        
        # Encontrar LRs que produjeron mejores pérdidas
        pairs = list(zip(
            self.memoria['historial_lr'][-100:],
            self.memoria['historial_loss'][-100:]
        ))
        
        if not pairs:
            return self.memoria['meta_params']['lr_optimo']
        
        # Ordenar por loss y tomar el promedio de los mejores
        pairs.sort(key=lambda x: x[1])
        mejores = pairs[:10]
        lr_optimo = np.mean([p[0] for p in mejores])
        
        self.memoria['meta_params']['lr_optimo'] = lr_optimo
        return lr_optimo
    
    def obtener_modulacion_optima(self) -> dict:
        """Obtiene el patrón de modulación promedio de los exitosos."""
        if len(self.memoria['patrones_talamo']) < 5:
            return None
        
        # Ordenar por loss
        patrones = sorted(
            self.memoria['patrones_talamo'],
            key=lambda x: x['loss']
        )[:20]  # Top 20
        
        # Promediar modulaciones
        modulacion_promedio = {}
        for key in patrones[0]['modulacion'].keys():
            valores = [p['modulacion'][key] for p in patrones]
            modulacion_promedio[key] = np.mean(valores)
        
        return modulacion_promedio


class AutoEntrenador:
    """
    Sistema de autoentrenamiento adaptativo.
    
    El modelo se autoajusta basado en:
    1. Historial de pérdidas
    2. Patrones exitosos guardados
    3. Meta-parámetros optimizados
    """
    
    def __init__(
        self,
        model: nn.Module,
        memoria: MemoriaAdaptativa,
        device: torch.device,
    ):
        self.model = model
        self.memoria = memoria
        self.device = device
        
        # Estado de autoajuste
        self.historial_loss = deque(maxlen=100)
        self.lr_actual = memoria.obtener_lr_optimo()
        self.paciencia = 0
        self.mejor_loss = float('inf')
        
        # Factores de ajuste
        self.factor_aumento_lr = 1.1
        self.factor_reduccion_lr = 0.5
        self.umbral_mejora = 0.001
        self.max_paciencia = 5
        
    def calcular_ajuste_lr(self, loss_actual: float) -> float:
        """
        Calcula el ajuste de learning rate basado en tendencia.
        
        Si loss está mejorando → mantener o aumentar lr ligeramente
        Si loss está estancado → reducir lr
        Si loss está empeorando → reducir lr significativamente
        """
        self.historial_loss.append(loss_actual)
        
        if len(self.historial_loss) < 10:
            return self.lr_actual
        
        # Calcular tendencia
        reciente = list(self.historial_loss)[-10:]
        tendencia = reciente[-1] - reciente[0]
        
        # Calcular varianza (estabilidad)
        varianza = np.var(reciente)
        
        if loss_actual < self.mejor_loss - self.umbral_mejora:
            # ¡Mejora! Mantener o aumentar ligeramente
            self.mejor_loss = loss_actual
            self.paciencia = 0
            if varianza < 0.01:  # Entrenamiento estable
                self.lr_actual *= self.factor_aumento_lr
        elif tendencia > 0:
            # Empeorando
            self.paciencia += 1
            if self.paciencia >= self.max_paciencia:
                self.lr_actual *= self.factor_reduccion_lr
                self.paciencia = 0
        
        # Limitar rango de LR
        self.lr_actual = max(1e-6, min(1e-2, self.lr_actual))
        
        # Registrar en memoria
        self.memoria.memoria['historial_lr'].append(self.lr_actual)
        self.memoria.memoria['historial_loss'].append(loss_actual)
        
        return self.lr_actual
    
    def ajustar_modulacion_talamo(self, stats: dict, loss: float):
        """
        Ajusta la temperatura/bias del tálamo basado en resultados.
        """
        # Extraer modulaciones actuales
        modulacion = {k: v for k, v in stats.items() if k.startswith('mod_')}
        
        if modulacion:
            self.memoria.registrar_patron_talamo(modulacion, loss)
        
        # Obtener modulación óptima
        mod_optima = self.memoria.obtener_modulacion_optima()
        
        return mod_optima
    
    def debe_guardar_checkpoint(self, loss: float) -> bool:
        """Decide si guardar checkpoint basado en historial."""
        if loss < self.mejor_loss:
            return True
        
        # También guardar si hay mejora significativa respecto a promedio
        if len(self.historial_loss) >= 10:
            promedio = np.mean(list(self.historial_loss)[-10:])
            if loss < promedio * 0.95:
                return True
        
        return False


class WikiTextDataset(Dataset):
    """Dataset para WikiText-103."""
    
    def __init__(self, data_path: str, seq_len: int = 128, max_tokens: int = None):
        self.seq_len = seq_len
        
        print(f'  📖 Cargando {data_path}...')
        tokens = []
        with open(data_path, 'r', encoding='utf-8') as f:
            while len(tokens) < (max_tokens or float('inf')):
                chunk = f.read(100000)
                if not chunk:
                    break
                tokens.extend([ord(c) % 256 for c in chunk])
        
        self.tokens = tokens[:max_tokens] if max_tokens else tokens
        print(f'  📊 Tokens cargados: {len(self.tokens):,}')
        
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def main():
    print('=' * 70)
    print('   🧠 LLARRI v7 - AUTOENTRENAMIENTO ADAPTATIVO')
    print('   El modelo aprende a aprender')
    print('=' * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n📱 Device: {device}')
    
    # === CONFIGURACIÓN INICIAL ===
    config = {
        'vocab_size': 256,
        'dim': 64,
        'n_heads': 2,
        'dropout': 0.15,
        'usar_hipocampo': False,
        'max_iteraciones': 1,
        'actividad_basal': 0.2,
        'seq_len': 128,
        'batch_size': 32,
        'train_tokens': 2_000_000,
        'val_tokens': 200_000,
        'gradient_clip': 0.5,
    }
    
    # === CARGAR MEMORIA ADAPTATIVA ===
    memoria = MemoriaAdaptativa()
    print(f'\n📚 Memoria Adaptativa:')
    print(f'   Entrenamientos previos: {memoria.memoria["estadisticas"]["total_entrenamientos"]}')
    print(f'   Mejor loss histórico: {memoria.memoria["estadisticas"]["mejor_loss_historico"]:.4f}')
    print(f'   LR óptimo aprendido: {memoria.obtener_lr_optimo():.2e}')
    
    # === CREAR MODELO ===
    print('\n🧠 Creando modelo LLARRI v7...')
    model = LLARRIv7Cerebral(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        usar_hipocampo=config['usar_hipocampo'],
        max_iteraciones=config['max_iteraciones'],
        actividad_basal=config['actividad_basal'],
    )
    model = model.to(device)
    
    # Cargar checkpoint si existe
    checkpoint_path = 'checkpoints/llarri_v7_auto_best.pt'
    if os.path.exists(checkpoint_path):
        print(f'   📂 Cargando checkpoint previo...')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f'   Continuando desde epoch {start_epoch}')
    else:
        start_epoch = 1
    
    print(f'   Parámetros: {sum(p.numel() for p in model.parameters()):,}')
    
    # === CARGAR DATOS ===
    print('\n📚 Cargando datos...')
    data_dir = Path('data/wikitext-103/wikitext-103-raw')
    
    train_dataset = WikiTextDataset(
        data_dir / 'wiki.train.raw',
        seq_len=config['seq_len'],
        max_tokens=config['train_tokens'],
    )
    
    val_dataset = WikiTextDataset(
        data_dir / 'wiki.valid.raw',
        seq_len=config['seq_len'],
        max_tokens=config['val_tokens'],
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # === CREAR AUTO-ENTRENADOR ===
    auto_entrenador = AutoEntrenador(model, memoria, device)
    
    # Usar LR aprendido de experiencias anteriores
    lr_inicial = memoria.obtener_lr_optimo()
    print(f'\n🎯 LR inicial (aprendido): {lr_inicial:.2e}')
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_inicial, weight_decay=0.01)
    
    # === ENTRENAMIENTO ADAPTATIVO ===
    print('\n' + '=' * 70)
    print('   INICIANDO AUTOENTRENAMIENTO')
    print('=' * 70)
    
    os.makedirs('checkpoints', exist_ok=True)
    max_epochs = 20
    mejor_val_loss = float('inf')
    
    for epoch in range(start_epoch, start_epoch + max_epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            result = model(x, targets=y)
            loss = result['loss']
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            # === AUTOAJUSTE DE LR ===
            if batch_idx % 100 == 0 and batch_idx > 0:
                nuevo_lr = auto_entrenador.calcular_ajuste_lr(loss.item())
                for param_group in optimizer.param_groups:
                    param_group['lr'] = nuevo_lr
            
            # Actualizar barra
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
            })
        
        train_loss /= max(train_steps, 1)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_steps = 0
        modulaciones_epoch = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                result = model(x, targets=y)
                
                if not torch.isnan(result['loss']):
                    val_loss += result['loss'].item()
                    val_steps += 1
                    
                    # Guardar modulaciones
                    stats = result.get('stats', {})
                    mod = {k: v for k, v in stats.items() if k.startswith('mod_')}
                    if mod:
                        modulaciones_epoch.append(mod)
        
        val_loss /= max(val_steps, 1)
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        
        # === REGISTRAR EN MEMORIA ===
        # Promediar modulaciones del epoch
        if modulaciones_epoch:
            mod_promedio = {}
            for key in modulaciones_epoch[0].keys():
                mod_promedio[key] = np.mean([m[key] for m in modulaciones_epoch])
            
            # Registrar patrón exitoso
            auto_entrenador.ajustar_modulacion_talamo({'stats': mod_promedio}, val_loss)
        
        # --- STATS ---
        lr_actual = optimizer.param_groups[0]['lr']
        
        print(f'\n📊 Epoch {epoch}:')
        print(f'   Train Loss: {train_loss:.4f}')
        print(f'   Val Loss:   {val_loss:.4f}')
        print(f'   Val PPL:    {val_ppl:.2f}')
        print(f'   LR actual:  {lr_actual:.2e}')
        print(f'   LR óptimo:  {memoria.obtener_lr_optimo():.2e}')
        
        # === GUARDAR SI ES MEJOR ===
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            
            # Registrar configuración exitosa
            memoria.registrar_exito(
                config={'epoch': epoch, 'lr': lr_actual},
                loss=val_loss,
                modulacion_talamo=mod_promedio if modulaciones_epoch else {},
            )
            
            # Guardar checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
                'config': config,
                'memoria_stats': memoria.memoria['estadisticas'],
            }, 'checkpoints/llarri_v7_auto_best.pt')
            
            print(f'   💾 ¡Mejor modelo guardado!')
            print(f'   🧠 Configuración registrada en memoria')
        
        # Guardar memoria periódicamente
        if epoch % 2 == 0:
            memoria.guardar()
            print(f'   💾 Memoria adaptativa guardada')
        
        # === EARLY STOPPING ADAPTATIVO ===
        if auto_entrenador.paciencia >= auto_entrenador.max_paciencia * 2:
            print(f'\n🛑 Early stopping adaptativo - Sin mejora significativa')
            break
    
    # === RESUMEN FINAL ===
    memoria.guardar()
    
    print('\n' + '=' * 70)
    print('   ✅ AUTOENTRENAMIENTO COMPLETADO')
    print('=' * 70)
    print(f'\n📊 Estadísticas de Memoria:')
    print(f'   Total entrenamientos: {memoria.memoria["estadisticas"]["total_entrenamientos"]}')
    print(f'   Mejor loss histórico: {memoria.memoria["estadisticas"]["mejor_loss_historico"]:.4f}')
    print(f'   LR óptimo aprendido:  {memoria.obtener_lr_optimo():.2e}')
    print(f'   Configs guardadas:    {len(memoria.memoria["configuraciones_exitosas"])}')
    
    # Mostrar modulación óptima aprendida
    mod_optima = memoria.obtener_modulacion_optima()
    if mod_optima:
        print(f'\n🎛️ Modulación Óptima Aprendida:')
        for nombre, valor in sorted(mod_optima.items()):
            barra = '█' * int(valor * 20) + '░' * (20 - int(valor * 20))
            print(f'   {nombre.replace("mod_", ""):12} {barra} {valor:.1%}')


if __name__ == '__main__':
    main()
