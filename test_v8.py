# Test rápido de LLARRI v8
from llarri_o1.cerebro import LLARRIv8, ConfigLLARRI
import torch

# Config LIVIANA para 4GB VRAM
config = ConfigLLARRI(
    vocab_size=8000, 
    dim=128,        # Reducido de 256
    n_capas=3,      # Reducido de 4
    n_heads=4,
    dropout=0.1,
    usar_axiomas=False,   # Desactivar para test rápido
    usar_memoria=False,   # Desactivar para test rápido
)
model = LLARRIv8(config)
print('Modelo creado!')

params = model.contar_parametros()
print(f'Total parametros: {params["total"]:,}')
for k, v in params.items():
    if k != 'total':
        print(f'  {k}: {v:,}')

# Test forward
x = torch.randint(0, 8000, (2, 64))
out = model(x)
print(f'Forward OK - logits shape: {out["logits"].shape}')
