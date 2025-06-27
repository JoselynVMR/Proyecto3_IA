import os
import torch
import pytorch_lightning as pl
import time

def get_model_size(path):
    """Devuelve el tamaño del archivo en MB."""
    return round(os.path.getsize(path) / 1e6, 2)

def measure_inference_time(model, dataloader, device, num_batches=100):
    """Mide el tiempo de inferencia promedio por lote."""
    model.eval()
    model.to(device)
    # Warm-up
    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = batch
            inputs = inputs.to(device)
            _ = model(inputs)
            break
    total_time = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs, _ = batch
            inputs = inputs.to(device)
            start_time = time.perf_counter()
            _ = model(inputs)
            total_time += time.perf_counter() - start_time
    avg_time = total_time / min(num_batches, len(dataloader))
    return avg_time * 1000  # Convertir a milisegundos

def evaluate_model(model, datamodule, accelerator, devices):
    """Evalúa el modelo usando PyTorch Lightning y devuelve un dict de métricas."""
    trainer = pl.Trainer(
        logger=False,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=False,
        enable_model_summary=False
    )
    results = trainer.test(model, datamodule=datamodule, verbose=False)
    return results[0]

def quantize_static_model(model, dataloader):

    model.cpu()
    model.eval()
    # Fusión de módulos
    fuse_list = [
        ['fc.1', 'fc.2'],  # Linear + BatchNorm1d
        ['fc.5', 'fc.6'],  # Linear + BatchNorm1d
        # Fusión para encoder1 (primer bloque)
        ['encoder.0.0', 'encoder.0.1'],  # Conv2d + BatchNorm2d
        ['encoder.0.3', 'encoder.0.4'],  # Conv2d + BatchNorm2d
        # Fusión para encoder2
        ['encoder.2.0', 'encoder.2.1'],
        ['encoder.2.3', 'encoder.2.4'],
        # Fusión para encoder3
        ['encoder.4.0', 'encoder.4.1'],
        ['encoder.4.3', 'encoder.4.4'],
        # Fusión para bottleneck
        ['encoder.6.0', 'encoder.6.1'],  # Conv2d + BatchNorm2d
        ['encoder.6.4', 'encoder.6.5']   # Conv2d + BatchNorm2d
    ]
    model = torch.quantization.fuse_modules(model, fuse_list, inplace=True)
    # Configurar cuantización
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    # Calibrar
    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = batch
            inputs = inputs.cpu()
            model(inputs)
    # Convertir
    torch.quantization.convert(model, inplace=True)
    return model