# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza
"""
LLARRI v8 - API Server

Servidor HTTP/WebSocket para inferencia con LLARRI v8.
Diseñado para escalabilidad en servidores con diferentes capacidades.

Uso local (desarrollo):
    python scripts/server.py

Uso en servidor:
    python scripts/server.py --host 0.0.0.0 --port 8080 --workers 4

Docker:
    docker run -p 8080:8080 -v ./checkpoints:/app/checkpoints llarri-server

Endpoints:
    POST /generate    - Genera texto a partir de un prompt
    POST /complete    - Completa texto (streaming)
    GET  /health      - Health check
    GET  /info        - Info del modelo
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

# Verificar dependencias opcionales
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    import torch
    import sentencepiece as spm
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Añadir path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

if HAS_FASTAPI:
    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: int = 100
        temperature: float = 0.8
        top_k: int = 50
        top_p: float = 0.9
        repetition_penalty: float = 1.2
        stream: bool = False
    
    class GenerateResponse(BaseModel):
        text: str
        prompt: str
        tokens_generated: int
        module_stats: dict = {}


# =============================================================================
# MODEL MANAGER
# =============================================================================

class ModelManager:
    """Gestiona el modelo LLARRI para inferencia."""
    
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        device: str = 'auto',
    ):
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        
        # Determinar device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.tokenizer = None
        self.config = None
        
    def load(self):
        """Carga el modelo y tokenizer."""
        from llarri_o1.config import LOCAL_4GB
        from llarri_o1.cerebro.model import LLARRIv8
        
        print(f"📥 Cargando modelo desde: {self.checkpoint_path}")
        print(f"   Device: {self.device}")
        
        # Cargar checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Configuración
        if 'config' in ckpt:
            self.config = ckpt['config']
        else:
            self.config = LOCAL_4GB
        
        # Modelo
        self.model = LLARRIv8(self.config).to(self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        
        # Tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(self.tokenizer_path)
        self.model.registrar_tokenizer(self.tokenizer)
        
        # Info
        params = self.model.contar_parametros()
        print(f"   Parámetros: {params['total']:,}")
        print(f"   Época: {ckpt.get('epoch', 'N/A')}")
        
        return self
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> tuple:
        """Genera texto."""
        # Tokenizar
        tokens = self.tokenizer.Encode(prompt)
        input_ids = torch.tensor([tokens], device=self.device)
        
        # Reset stats
        self.model.reset_estadisticas()
        
        # Generar
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        
        # Decodificar
        text = self.tokenizer.Decode(output[0].tolist())
        
        # Stats
        stats = self.model.obtener_estadisticas_modulos()
        
        return text, len(output[0]) - len(tokens), stats
    
    @torch.no_grad()
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ):
        """Genera texto en streaming."""
        import torch.nn.functional as F
        
        # Tokenizar
        tokens = self.tokenizer.Encode(prompt)
        generated = torch.tensor([tokens], device=self.device)
        
        # Yield prompt primero
        yield prompt
        
        for _ in range(max_tokens):
            # Truncar si excede max_seq_len
            if generated.shape[1] >= self.config.max_seq_len:
                context = generated[:, -self.config.max_seq_len:]
            else:
                context = generated
            
            # Forward
            outputs = self.model(context)
            logits = outputs['logits'][:, -1, :]
            
            # Repetition penalty
            for token_id in set(generated[0].tolist()):
                logits[0, token_id] /= repetition_penalty
            
            # Temperature
            logits = logits / temperature
            
            # Top-k
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Decode y yield
            token_text = self.tokenizer.Decode([next_token[0].item()])
            yield token_text
            
            # Stop en EOS
            if next_token[0].item() == 3:
                break
            
            # Pequeña pausa para streaming
            await asyncio.sleep(0.01)
    
    def get_info(self) -> dict:
        """Retorna información del modelo."""
        params = self.model.contar_parametros() if self.model else {}
        
        return {
            'name': 'LLARRI v8',
            'version': '8.0.0',
            'parameters': params.get('total', 0),
            'device': str(self.device),
            'config': {
                'dim': getattr(self.config, 'dim', 0),
                'n_capas': getattr(self.config, 'n_capas', 0),
                'n_heads': getattr(self.config, 'n_heads', 0),
                'vocab_size': getattr(self.config, 'vocab_size', 0),
                'max_seq_len': getattr(self.config, 'max_seq_len', 0),
            } if self.config else {},
        }


# =============================================================================
# FASTAPI APP
# =============================================================================

def create_app(model_manager: ModelManager) -> 'FastAPI':
    """Crea la aplicación FastAPI."""
    app = FastAPI(
        title="LLARRI v8 API",
        description="API para generación de texto con LLARRI v8",
        version="8.0.0",
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup():
        model_manager.load()
    
    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": model_manager.model is not None}
    
    @app.get("/info")
    async def info():
        return model_manager.get_info()
    
    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        if model_manager.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if request.stream:
            async def stream_generator():
                async for token in model_manager.generate_stream(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                ):
                    yield token
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/plain",
            )
        
        text, tokens_generated, stats = model_manager.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )
        
        return GenerateResponse(
            text=text,
            prompt=request.prompt,
            tokens_generated=tokens_generated,
            module_stats=stats,
        )
    
    return app


# =============================================================================
# SIMPLE HTTP SERVER (sin FastAPI)
# =============================================================================

def run_simple_server(model_manager: ModelManager, host: str, port: int):
    """Servidor HTTP simple sin dependencias extra."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    
    model_manager.load()
    
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/health':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode())
            
            elif self.path == '/info':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(model_manager.get_info()).encode())
            
            else:
                self.send_response(404)
                self.end_headers()
        
        def do_POST(self):
            if self.path == '/generate':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                request = json.loads(post_data)
                
                text, tokens, stats = model_manager.generate(
                    prompt=request.get('prompt', ''),
                    max_tokens=request.get('max_tokens', 100),
                    temperature=request.get('temperature', 0.8),
                    top_k=request.get('top_k', 50),
                    top_p=request.get('top_p', 0.9),
                    repetition_penalty=request.get('repetition_penalty', 1.2),
                )
                
                response = {
                    'text': text,
                    'prompt': request.get('prompt', ''),
                    'tokens_generated': tokens,
                    'module_stats': stats,
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            
            else:
                self.send_response(404)
                self.end_headers()
    
    server = HTTPServer((host, port), Handler)
    print(f"🚀 Servidor iniciado en http://{host}:{port}")
    print("   Endpoints: /health, /info, /generate")
    server.serve_forever()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='LLARRI v8 Server')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/llarri_v8_best.pt',
                       help='Path al checkpoint')
    parser.add_argument('--tokenizer', type=str,
                       default='data/tokenizer/llarri_bpe.model',
                       help='Path al tokenizer')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host (0.0.0.0 para acceso externo)')
    parser.add_argument('--port', type=int, default=8080,
                       help='Puerto')
    parser.add_argument('--workers', type=int, default=1,
                       help='Número de workers (solo FastAPI)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda)')
    parser.add_argument('--simple', action='store_true',
                       help='Usar servidor HTTP simple (sin FastAPI)')
    args = parser.parse_args()
    
    # Verificar archivos
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint no encontrado: {args.checkpoint}")
        return
    
    if not os.path.exists(args.tokenizer):
        print(f"❌ Tokenizer no encontrado: {args.tokenizer}")
        return
    
    # Crear model manager
    model_manager = ModelManager(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )
    
    print("\n" + "=" * 60)
    print("LLARRI v8 - SERVER")
    print("=" * 60)
    
    if args.simple or not HAS_FASTAPI:
        if not HAS_FASTAPI:
            print("⚠️ FastAPI no instalado, usando servidor simple")
            print("   Para instalar: pip install fastapi uvicorn")
        run_simple_server(model_manager, args.host, args.port)
    else:
        app = create_app(model_manager)
        print(f"\n🚀 Iniciando servidor FastAPI...")
        print(f"   URL: http://{args.host}:{args.port}")
        print(f"   Docs: http://{args.host}:{args.port}/docs")
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers if args.workers > 1 else None,
        )


if __name__ == '__main__':
    main()
