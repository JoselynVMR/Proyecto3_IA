import os
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
import time
import sys
import wandb

# 📁 Agregar ruta raíz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 🧠 Importar clases del proyecto
from scripts.models.classifier_quantizable import QuantizableButterflyClassifier
from scripts.data_module import DataModule

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_pct", type=int, default=30,
                        help="Porcentaje de datos etiquetados (ej. 30 para 0.3)")
    args = parser.parse_args()

    # Hiperparámetros
    hparams = {
        'batch_size': 64,
        'num_workers': 4,
        'seed': 42,
        'learning_rate': 1e-3,
        'label_pct': args.label_pct / 100.0,
        'num_classes': 30
    }

    # Configurar clave API de WandB
    #os.environ["WANDB_API_KEY"] = "757af0e5727478d40e4a586ed9175f733ee00948"
    os.environ["WANDB_API_KEY"] = "3e7282c2a62557882828c8d06b01ec4b8f7135a1"    # Llave Joselyn

    # Inicializar WandB
    wandb.init(
        project="butterfly-classification",
        config=hparams,
        name=f"eval_{args.label_pct}pct",
        group="model_evaluation"
    )

    # 📁 Rutas base
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.abspath(os.path.join(current_dir, f"../../checkpoints/classifier/cs{args.label_pct}"))
    data_dir = os.path.abspath(os.path.join(current_dir, "../../data/species_selected"))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    classifiers = ["Classifier_A", "Classifier_B1", "Classifier_B2"]
    table_data = []

    for name in classifiers:
        print(f"\n=== {name} ({args.label_pct}% etiquetado) ===")
        ckpt_path = os.path.join(ckpt_dir, f"best-{name}.ckpt")
        if not os.path.isfile(ckpt_path):
            print(f"❌ Checkpoint no encontrado: {ckpt_path}")
            continue

        # Cargar modelo desde checkpoint
        model_orig = QuantizableButterflyClassifier.load_from_checkpoint(
            ckpt_path,
            hparams=hparams,
            map_location=device
        )
        model_orig.eval()

        # Configurar DataModule
        datamodule = DataModule(hparams=hparams, data_dir=data_dir)
        datamodule.setup(stage="test")
        test_dataloader = datamodule.test_dataloader()

        # Evaluar modelo original
        size_orig = get_model_size(ckpt_path)
        latency_orig = measure_inference_time(model_orig, test_dataloader, device)
        metrics_orig = evaluate_model(
            model_orig, datamodule,
            accelerator="gpu" if use_cuda else "cpu",
            devices=1
        )
        print(f"📦 Tamaño original: {size_orig} MB")
        print(f"⏱ Latencia original: {latency_orig:.2f} ms por lote")
        print(f"   🔹 test_loss:      {metrics_orig.get('test_loss', float('nan')):.4f}")
        print(f"   🔹 test_acc:       {metrics_orig.get('test_acc', float('nan')):.4f}")
        print(f"   🔹 test_precision: {metrics_orig.get('test_precision', float('nan')):.4f}")
        print(f"   🔹 test_recall:    {metrics_orig.get('test_recall', float('nan')):.4f}")
        print(f"   🔹 test_f1:        {metrics_orig.get('test_f1', float('nan')):.4f}")

        # Registrar métricas en WandB
        wandb.log({
            f"{name}_original/test_loss": metrics_orig.get('test_loss', float('nan')),
            f"{name}_original/test_acc": metrics_orig.get('test_acc', float('nan')),
            f"{name}_original/test_precision": metrics_orig.get('test_precision', float('nan')),
            f"{name}_original/test_recall": metrics_orig.get('test_recall', float('nan')),
            f"{name}_original/test_f1": metrics_orig.get('test_f1', float('nan')),
            f"{name}_original/size_mb": size_orig,
            f"{name}_original/latency_ms": latency_orig
        })

        # Cuantización estática (cargar modelo original nuevamente para evitar conflictos)
        model_orig_static = QuantizableButterflyClassifier.load_from_checkpoint(
            ckpt_path,
            hparams=hparams,
            map_location=torch.device("cpu")
        )
        quant_static_model = quantize_static_model(model_orig_static, test_dataloader)
        qpth_static = ckpt_path.replace(".ckpt", "_quantized_static.pth")
        torch.save(quant_static_model.state_dict(), qpth_static)

        # Evaluar modelo cuantizado estático
        size_quant_static = get_model_size(qpth_static)
        latency_quant_static = measure_inference_time(quant_static_model, test_dataloader, torch.device("cpu"))
        metrics_quant_static = evaluate_model(
            quant_static_model, datamodule,
            accelerator="cpu",
            devices=1
        )
        print(f"📦 Tamaño cuantizado estático: {size_quant_static} MB")
        print(f"⏱ Latencia cuantizada estática: {latency_quant_static:.2f} ms por lote")
        print(f"   🔹 test_loss:      {metrics_quant_static.get('test_loss', float('nan')):.4f}")
        print(f"   🔹 test_acc:       {metrics_quant_static.get('test_acc', float('nan')):.4f}")
        print(f"   🔹 test_precision: {metrics_quant_static.get('test_precision', float('nan')):.4f}")
        print(f"   🔹 test_recall:    {metrics_quant_static.get('test_recall', float('nan')):.4f}")
        print(f"   🔹 test_f1:        {metrics_quant_static.get('test_f1', float('nan')):.4f}")

        # Registrar métricas estáticas en WandB
        wandb.log({
            f"{name}_quantized_static/test_loss": metrics_quant_static.get('test_loss', float('nan')),
            f"{name}_quantized_static/test_acc": metrics_quant_static.get('test_acc', float('nan')),
            f"{name}_quantized_static/test_precision": metrics_quant_static.get('test_precision', float('nan')),
            f"{name}_quantized_static/test_recall": metrics_quant_static.get('test_recall', float('nan')),
            f"{name}_quantized_static/test_f1": metrics_quant_static.get('test_f1', float('nan')),
            f"{name}_quantized_static/size_mb": size_quant_static,
            f"{name}_quantized_static/latency_ms": latency_quant_static
        })

        # Agregar datos a la tabla
        table_data.append([
            name, args.label_pct,
            metrics_orig.get('test_loss', float('nan')),
            metrics_orig.get('test_acc', float('nan')),
            metrics_orig.get('test_precision', float('nan')),
            metrics_orig.get('test_recall', float('nan')),
            metrics_orig.get('test_f1', float('nan')),
            metrics_quant_static.get('test_loss', float('nan')),
            metrics_quant_static.get('test_acc', float('nan')),
            metrics_quant_static.get('test_precision', float('nan')),
            metrics_quant_static.get('test_recall', float('nan')),
            metrics_quant_static.get('test_f1', float('nan'))
        ])

    # Crear y registrar tabla en WandB
    columns = [
        "Modelo", "Porcentaje Etiquetado",
        "Tamaño Original (MB)", "Tamaño Cuantizado Estático (MB)",
        "Latencia Original (ms)", "Latencia Cuantizada Estática (ms)",
        "Loss Original", "Acc Original", "Precision Original", "Recall Original", "F1 Original",
        "Loss Cuantizado Estático", "Acc Cuantizado Estático", "Precision Cuantizado Estático",
        "Recall Cuantizado Estático", "F1 Cuantizado Estático"
    ]
    table = wandb.Table(data=table_data, columns=columns)
    wandb.log({"Model_Comparison": table})
    wandb.finish()

    # Imprimir resumen para informe
    print("\n=== Resumen===")
    for row in table_data:
        print(f"\nModelo: {row[0]} ({row[1]}% etiquetado)")
        print(f"Tamaño: Original={row[2]:.2f} MB, Cuantizado={row[4]:.2f} MB")
        print(f"Latencia: Original={row[5]:.2f} ms, Cuantizado={row[7]:.2f} ms")
        print(f"Métricas Originales: Loss={row[8]:.4f}, Acc={row[9]:.4f}, Precision={row[10]:.4f}, Recall={row[11]:.4f}, F1={row[12]:.4f}")
        print(f"Métricas Cuantizadas: Loss={row[18]:.4f}, Acc={row[19]:.4f}, Precision={row[20]:.4f}, Recall={row[21]:.4f}, F1={row[22]:.4f}")

if __name__ == "__main__":
    main()