# Test rápido de LLARRI v8
from llarri_o1.cerebro import LLARRIv8, ConfigLLARRI
import torch

# Config LIVIANA para 4GB VRAM
config = ConfigLLARRI(
    vocab_size=8000, 
    dim=128,        # Reducido
    n_capas=3,      # Reducido
    n_heads=4,
    dropout=0.1,
    usar_axiomas=True,    # Activado
    usar_memoria=True,    # Activado
    capacidad_memoria=100,  # Reducido
)
model = LLARRIv8(config)
print('Modelo LLARRI v8 creado!')

params = model.contar_parametros()
print(f'Total parametros: {params["total"]:,}')
for k, v in params.items():
    if k != 'total' and v > 0:
        print(f'  {k}: {v:,}')

# Test forward
x = torch.randint(0, 8000, (2, 64))
out = model(x)
print(f'Forward OK - logits shape: {out["logits"].shape}')

# Test con labels (para training)
y = torch.randint(0, 8000, (2, 64))
out = model(x, labels=y)
print(f'Loss: {out["loss"].item():.4f}')

# Test generación
print('\nTest generación:')
prompt = torch.tensor([[1, 2, 3, 4, 5]])  # Tokens de ejemplo
generated = model.generate(prompt, max_new_tokens=10)
print(f'Generated shape: {generated.shape}')
