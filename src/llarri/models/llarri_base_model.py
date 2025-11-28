import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from transformers import TrOCRProcessor, AutoTokenizer

from .encoder_vit import ViTEncoder, ViTEncoderConfig
from .decoder_trocr import TrOCRDecoder, TrOCRDecoderConfig

# Simple character-level vocabulary (backup, but we'll use HuggingFace tokenizer)
VOCAB = (
    list("abcdefghijklmnopqrstuvwxyz")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("0123456789")
    + list(".,!?:;\"'()-")
    + ["<pad>", "<sos>", "<eos>"]
)
CHAR2ID = {c: i for i, c in enumerate(VOCAB)}
ID2CHAR = {i: c for i, c in enumerate(VOCAB)}

def tokenize(text: str) -> List[int]:
    """Convert a string into a list of token IDs.
    Unknown characters are mapped to the <pad> token.
    """
    return [CHAR2ID.get(ch, CHAR2ID["<pad>"]) for ch in text]

def detokenize(ids: List[int]) -> str:
    """Convert a list of token IDs back into a string, ignoring special tokens."""
    special = {CHAR2ID["<pad>"], CHAR2ID["<sos>"], CHAR2ID["<eos>"]}
    chars = [ID2CHAR.get(i, "") for i in ids if i not in special]
    return "".join(chars)

class LlarriBaseModel(pl.LightningModule):
    def __init__(
        self,
        encoder_cfg: Optional[ViTEncoderConfig] = None,
        decoder_cfg: Optional[TrOCRDecoderConfig] = None,
        vocab: Optional[List[str]] = None,
        learning_rate: float = 1e-4,
        use_custom_vocab: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder_cfg = encoder_cfg or ViTEncoderConfig()
        self.decoder_cfg = decoder_cfg or TrOCRDecoderConfig()
        self.learning_rate = learning_rate
        self.use_custom_vocab = use_custom_vocab
        
        # Initialize encoder and decoder
        self.encoder = ViTEncoder(self.encoder_cfg)
        self.decoder = TrOCRDecoder(self.decoder_cfg)
        
        # Initialize processor and tokenizer
        # Using TrOCR's pretrained processor for proper tokenization
        try:
            self.processor = TrOCRProcessor.from_pretrained(
                self.decoder_cfg.pretrained_model_name
            )
            self.tokenizer = self.processor.tokenizer
        except:
            # Fallback to standalone tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.decoder_cfg.pretrained_model_name
            )
            self.processor = None
        
        # Vocabulary handling (keep for compatibility, but prefer HuggingFace tokenizer)
        if vocab is not None:
            self.vocab = vocab
            self.char2id = {c: i for i, c in enumerate(vocab)}
            self.id2char = {i: c for i, c in enumerate(vocab)}
        else:
            self.vocab = VOCAB
            self.char2id = CHAR2ID
            self.id2char = ID2CHAR

    def forward(
        self, 
        pixel_values: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
    ):
        """Run encoder and decoder.
        
        Args:
            pixel_values: Images tensor (B, C, H, W) - preprocessed by ViT
            labels: Optional token IDs for teacher forcing and loss computation
            decoder_input_ids: Optional decoder input for custom decoding
        
        Returns:
            Decoder output with loss if labels provided
        """
        # Encode images
        encoder_hidden = self.encoder(pixel_values)
        
        # Decode with labels for training
        decoder_output = self.decoder(
            encoder_hidden_states=encoder_hidden, 
            labels=labels
        )
        return decoder_output

    def generate(
        self, 
        pixel_values: torch.Tensor, 
        max_length: int = 128, 
        num_beams: int = 4,
        **gen_kwargs
    ):
        """Auto-regressive generation.
        
        Args:
            pixel_values: Images tensor (B, C, H, W)
            max_length: Maximum sequence length
            num_beams: Number of beams for beam search
            **gen_kwargs: Additional generation arguments
        
        Returns:
            List of decoded strings (one per batch element)
        """
        encoder_hidden = self.encoder(pixel_values)
        generated_ids = self.decoder.generate(
            encoder_hidden_states=encoder_hidden, 
            max_length=max_length,
            num_beams=num_beams,
            **gen_kwargs
        )
        
        # Decode using tokenizer
        if self.tokenizer:
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            # Fallback to custom detokenize
            return [detokenize(ids.tolist()) for ids in generated_ids]

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """Training step - works with dictionary batches from DataModule."""
        # Extract from batch dictionary
        pixel_values = batch["pixel_values"]  # (B, C, H, W)
        labels = batch["labels"]  # (B, seq_len) - tokenized text
        
        # Forward pass
        output = self(pixel_values, labels=labels)
        loss = output.loss if hasattr(output, "loss") else output["loss"]
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step - works with dictionary batches from DataModule."""
        # Extract from batch dictionary
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        
        # Forward pass
        output = self(pixel_values, labels=labels)
        loss = output.loss if hasattr(output, "loss") else output["loss"]
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Optional: Generate predictions and compute metrics
        if batch_idx % 10 == 0:  # Sample every 10 batches
            with torch.no_grad():
                predictions = self.generate(pixel_values, max_length=64)
                # Could compute CER/WER here if needed
        
        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Optional: Add learning rate scheduler
        # Note: 'verbose' argument removed in PyTorch 2.x
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def predict(
        self,
        image,
        preprocess: bool = True,
        use_opencv: bool = True,
        use_language_model: bool = True,
        use_ensemble: bool = False,
        ensemble_strategy: str = "verify_if_low_conf",
        return_metadata: bool = False,
        max_length: int = 128,
        num_beams: int = 4,
    ):
        """
        Predice texto desde una imagen con preprocesamiento automático.
        
        Este método integra:
        1. Preprocesador (equivalente a imageprocessor.js)
        2. Modelo de lenguaje (cadenas de Markov + reglas fonéticas + spellcheck)
        3. Ensemble con MiniCPM-V (opcional, para máxima precisión)
        
        El modelo de lenguaje mejora la precisión aplicando:
        - N-gramas de caracteres para detectar secuencias improbables
        - Reglas fonéticas del español (Q siempre con U, etc.)
        - Corrector ortográfico con diccionario español
        
        El ensemble (si está activado) combina:
        - LLARRI (TrOCR especializado)
        - MiniCPM-V (modelo multimodal poderoso)
        - SpanishLanguageModel (post-procesamiento)
        
        Args:
            image: Puede ser:
                - str/Path: ruta a imagen
                - PIL.Image: imagen PIL
                - np.ndarray: array numpy (H, W, C) o (H, W)
                - torch.Tensor: tensor ya preprocesado (C, H, W) o (B, C, H, W)
            preprocess: Si aplicar preprocesamiento automático
            use_opencv: Usar OpenCV (mejor) o Pillow (más compatible)
            use_language_model: Si aplicar modelo de lenguaje para corregir
            use_ensemble: Si usar ensemble con MiniCPM-V (más lento pero más preciso)
            ensemble_strategy: Estrategia de ensemble:
                - "verify_if_low_conf": Solo verifica si LLARRI tiene baja confianza
                - "always_verify": Siempre verifica con MiniCPM
                - "consensus": Acepta si múltiples modelos coinciden
                - "rerank": Genera candidatos y re-rankea con LM
                - "llarri_only": Solo LLARRI + LM
                - "minicpm_only": Solo MiniCPM + LM
            return_metadata: Si retornar metadata adicional (confianza, timing, etc.)
            max_length: Longitud máxima del texto generado
            num_beams: Beams para beam search
            
        Returns:
            str: Texto reconocido (si return_metadata=False)
            tuple: (texto, metadata) si return_metadata=True
        """
        # Si ensemble está activo, delegar al EnsembleOCR
        if use_ensemble:
            return self._predict_with_ensemble(
                image,
                strategy=ensemble_strategy,
                return_metadata=return_metadata,
            )
        from ..inference.preprocess_opencv import preprocess_to_tensor, preprocess_simple_to_tensor
        
        self.eval()
        device = next(self.parameters()).device
        
        # Si ya es un tensor preprocesado
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
            pixel_values = image.to(device)
        else:
            # Preprocesar imagen
            if preprocess:
                if use_opencv:
                    try:
                        pixel_values = preprocess_to_tensor(
                            image,
                            target_size=(224, 224),
                            device=str(device),
                            use_opencv=True
                        )
                    except:
                        # Fallback a versión simple
                        pixel_values = preprocess_simple_to_tensor(
                            image,
                            target_size=(224, 224),
                            device=str(device)
                        )
                else:
                    pixel_values = preprocess_simple_to_tensor(
                        image,
                        target_size=(224, 224),
                        device=str(device)
                    )
            else:
                # Sin preprocesamiento, usar processor de HuggingFace
                from PIL import Image as PILImage
                if isinstance(image, str):
                    image = PILImage.open(image).convert("RGB")
                elif isinstance(image, np.ndarray):
                    image = PILImage.fromarray(image).convert("RGB")
                
                pixel_values = self.processor(image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(device)
        
        # Generar texto
        with torch.no_grad():
            results = self.generate(
                pixel_values,
                max_length=max_length,
                num_beams=num_beams
            )
        
        # Aplicar modelo de lenguaje para corregir
        if use_language_model:
            try:
                from ..inference.language_model import get_language_model
                lm = get_language_model()
                results = [lm.correct_text(r) for r in results]
            except ImportError:
                pass  # Si no están las dependencias, continuar sin LM
        
        # Retornar primer resultado (o lista si batch)
        if len(results) == 1:
            return results[0]
        return results

    def _predict_with_ensemble(
        self,
        image,
        strategy: str = "verify_if_low_conf",
        return_metadata: bool = False,
    ):
        """
        Predicción usando el ensemble completo.
        
        Args:
            image: Imagen a procesar
            strategy: Estrategia de ensemble
            return_metadata: Si retornar metadata
            
        Returns:
            str o tuple(str, dict) según return_metadata
        """
        from ..inference.ensemble_ocr import EnsembleOCR, EnsembleStrategy, EnsembleConfig
        
        # Crear ensemble con este modelo como LLARRI
        config = EnsembleConfig(
            strategy=EnsembleStrategy(strategy),
            use_language_model=True,
        )
        
        ensemble = EnsembleOCR(
            llarri_model=self,
            config=config,
        )
        
        # Predecir
        result = ensemble.predict(image)
        
        if return_metadata:
            return result.text, result.to_dict()
        return result.text

    def predict_batch(
        self,
        images: list,
        preprocess: bool = True,
        use_language_model: bool = True,
        **kwargs
    ) -> list:
        """
        Predice texto para múltiples imágenes.
        
        Args:
            images: Lista de imágenes (paths, PIL, numpy, etc.)
            preprocess: Si aplicar preprocesamiento
            use_language_model: Si aplicar modelo de lenguaje
            **kwargs: Argumentos adicionales para predict
            
        Returns:
            Lista de textos reconocidos
        """
        return [
            self.predict(
                img, 
                preprocess=preprocess, 
                use_language_model=use_language_model,
                **kwargs
            ) 
            for img in images
        ]

    def export_onnx(self, sample_image: torch.Tensor, onnx_path: str):
        """Export the full model (encoder + decoder) to ONNX.
        
        Args:
            sample_image: Single image tensor (C, H, W)
            onnx_path: Path to save ONNX model
        """
        self.eval()
        dummy_input = sample_image.unsqueeze(0)  # (1, C, H, W)
        
        torch.onnx.export(
            self,
            dummy_input,
            onnx_path,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={
                "pixel_values": {0: "batch"}, 
                "logits": {0: "batch", 1: "sequence"}
            },
            opset_version=14,
        )
        print(f"Model exported to {onnx_path}")
