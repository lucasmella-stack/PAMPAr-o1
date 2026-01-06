import gradio as gr
import torch
import os
from huggingface_hub import login

# ============ CONFIGURACIÓN ============
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

def train_model(epochs, batch_size, learning_rate, start_sample, num_samples):
    """Entrena el modelo con un rango específico de muestras"""
    
    yield "🚀 Iniciando entrenamiento LLARRI...\n"
    yield f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n"
    
    if torch.cuda.is_available():
        yield f"✅ GPU: {torch.cuda.get_device_name(0)}\n"
    else:
        yield "⚠️ Sin GPU, usando CPU (muy lento)\n"
    
    if not HF_TOKEN:
        yield "❌ Error: Configura HF_TOKEN en Settings > Secrets"
        return
    yield "✅ Token OK\n"
    
    # Cargar dataset
    yield "📂 Cargando dataset...\n"
    from datasets import load_dataset
    dataset = load_dataset("lucas-mella/llarri-spanish-htr", split="train")
    total_samples = len(dataset)
    yield f"✅ {total_samples} muestras totales\n"
    
    # Seleccionar rango
    end_sample = min(start_sample + num_samples, total_samples)
    actual_samples = end_sample - start_sample
    
    if start_sample >= total_samples:
        yield f"❌ Error: start_sample ({start_sample}) >= total ({total_samples})"
        return
    
    dataset = dataset.select(range(start_sample, end_sample))
    yield f"📊 Usando muestras {start_sample} a {end_sample} ({actual_samples} muestras)\n"
    
    # Cargar modelo (continuar desde el entrenado)
    yield "🔧 Cargando modelo pre-entrenado...\n"
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    
    # Intentar cargar nuestro modelo, si no existe usar el base
    try:
        model = VisionEncoderDecoderModel.from_pretrained("lucas-mella/llarri-spanish-htr-model")
        yield "✅ Modelo LLARRI cargado (continuando entrenamiento)\n"
    except:
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        yield "✅ Modelo base cargado (primer entrenamiento)\n"
    
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    
    # Preprocesar
    yield "🔄 Preprocesando datos...\n"
    from PIL import Image
    import io
    
    def preprocess(examples):
        images = []
        for img_data in examples["image"]:
            if isinstance(img_data, dict) and "bytes" in img_data:
                img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
            elif isinstance(img_data, Image.Image):
                img = img_data.convert("RGB")
            else:
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
            images.append(img)
        
        pixel_values = processor(images, return_tensors="pt").pixel_values
        labels = processor.tokenizer(
            examples["text"],
            padding="max_length",
            max_length=128,
            truncation=True
        ).input_ids
        
        labels = [[-100 if token == processor.tokenizer.pad_token_id else token for token in label] for label in labels]
        
        return {"pixel_values": pixel_values, "labels": labels}
    
    dataset = dataset.map(preprocess, batched=True, batch_size=32, remove_columns=dataset.column_names)
    yield f"✅ {len(dataset)} muestras preprocesadas\n"
    
    # Data collator
    class DataCollatorForTrOCR:
        def __init__(self, processor):
            self.processor = processor
        
        def __call__(self, features):
            pixel_values = torch.stack([torch.tensor(f["pixel_values"]) if not isinstance(f["pixel_values"], torch.Tensor) else f["pixel_values"] for f in features])
            
            labels_list = [f["labels"] if isinstance(f["labels"], list) else f["labels"].tolist() for f in features]
            max_len = max(len(l) for l in labels_list)
            padded = [l + [-100] * (max_len - len(l)) for l in labels_list]
            labels = torch.tensor(padded)
            
            return {"pixel_values": pixel_values, "labels": labels}
    
    # Entrenar
    yield f"🏋️ Entrenando {actual_samples} muestras por {epochs} epochs...\n"
    
    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./output",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        save_strategy="no",
        logging_steps=50,
        remove_unused_columns=False,
        predict_with_generate=True,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForTrOCR(processor),
    )
    
    trainer.train()
    yield "✅ Entrenamiento completado!\n"
    
    # Guardar
    yield "💾 Subiendo modelo actualizado...\n"
    model.push_to_hub("lucas-mella/llarri-spanish-htr-model", token=HF_TOKEN)
    processor.push_to_hub("lucas-mella/llarri-spanish-htr-model", token=HF_TOKEN)
    
    yield f"""
🎉 ¡Entrenamiento completado!
📊 Muestras usadas: {start_sample} - {end_sample}
📦 Modelo: https://huggingface.co/lucas-mella/llarri-spanish-htr-model
"""

# Interfaz
with gr.Blocks(title="LLARRI Trainer v2") as demo:
    gr.Markdown("# 🇪🇸 LLARRI Spanish HTR - Entrenamiento Incremental")
    gr.Markdown("Entrena el modelo con rangos específicos del dataset")
    
    with gr.Row():
        with gr.Column():
            epochs = gr.Slider(1, 5, value=3, step=1, label="Epochs")
            batch_size = gr.Slider(1, 8, value=4, step=1, label="Batch Size")
            lr = gr.Number(value=5e-5, label="Learning Rate")
        with gr.Column():
            start_sample = gr.Number(value=10000, label="Muestra inicial", precision=0)
            num_samples = gr.Number(value=5000, label="Cantidad de muestras", precision=0)
    
    btn = gr.Button("🚀 Entrenar", variant="primary")
    output = gr.Textbox(label="Progreso", lines=20)
    
    btn.click(train_model, [epochs, batch_size, lr, start_sample, num_samples], output)

demo.launch()
