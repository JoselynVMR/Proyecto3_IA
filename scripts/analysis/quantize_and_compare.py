import os
import sys
import torch
import pytorch_lightning as pl
import wandb
from hydra import main
from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.models.classifier_quantizable import QuantizableButterflyClassifier
from scripts.data_module import DataModule
from scripts.evaluatin_utils import get_model_size, measure_inference_time, evaluate_model, quantize_static_model

GlobalHydra.instance().clear()

@main(config_path='../../configuration', config_name='config', version_base=None)
def quantize_and_compare(cfg: DictConfig):

    os.environ["WANDB_API_KEY"] = cfg.key.api_key

    label_pct = cfg.experiment.train_classifier.datamodule.label_pct
    label = cfg.experiment.train_classifier.datamodule.label
    classifiers = cfg.experiment.train_classifier.classifiers
    data_dir = cfg.experiment.train_classifier.paths.data_dir
    ckpt_dir = cfg.experiment.train_classifier.paths.checkpoint_dir

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    table_data = []

    for config in classifiers:
        name = config.name
        run_name = f"{name}_{label}pct_eval"
        wandb_logger = instantiate(cfg.experiment.train_classifier.wandb, name=run_name, group="quantization_evaluation")

        print(f"\n=== {name} ({label}% etiquetado) ===")

        ckpt_path = os.path.join(ckpt_dir, f"best-{name}.ckpt")
        if not os.path.isfile(ckpt_path):
            print(f"❌ Checkpoint no encontrado: {ckpt_path}")
            continue

        # Modelo original
        model_orig = QuantizableButterflyClassifier.load_from_checkpoint(
            ckpt_path,
            hparams=dict(cfg.experiment.train_classifier.datamodule),
            map_location=device
        )
        model_orig.eval()

        datamodule = DataModule(cfg.experiment.train_classifier.datamodule, data_dir)
        datamodule.setup(stage="test")
        test_loader = datamodule.test_dataloader()

        size_orig = get_model_size(ckpt_path)
        latency_orig = measure_inference_time(model_orig, test_loader, device)
        metrics_orig = evaluate_model(model_orig, datamodule, accelerator="gpu" if use_cuda else "cpu", devices=1)

        wandb_logger.experiment.log({
            f"{name}_original/test_loss": metrics_orig.get('test_loss', float('nan')),
            f"{name}_original/test_acc": metrics_orig.get('test_acc', float('nan')),
            f"{name}_original/test_precision": metrics_orig.get('test_precision', float('nan')),
            f"{name}_original/test_recall": metrics_orig.get('test_recall', float('nan')),
            f"{name}_original/test_f1": metrics_orig.get('test_f1', float('nan')),
            f"{name}_original/size_mb": size_orig,
            f"{name}_original/latency_ms": latency_orig
        })

        # Cuantización
        model_static = QuantizableButterflyClassifier.load_from_checkpoint(
            ckpt_path,
            hparams=dict(cfg.experiment.train_classifier.datamodule),
            map_location=torch.device("cpu")
        )
        model_static.eval()

        quant_model = quantize_static_model(model_static, test_loader)
        quant_path = ckpt_path.replace(".ckpt", "_quantized_static.pth")
        torch.save(quant_model.state_dict(), quant_path)

        size_quant = get_model_size(quant_path)
        latency_quant = measure_inference_time(quant_model, test_loader, torch.device("cpu"))
        metrics_quant = evaluate_model(quant_model, datamodule, accelerator="cpu", devices=1)

        wandb_logger.experiment.log({
            f"{name}_quant/test_loss": metrics_quant.get('test_loss', float('nan')),
            f"{name}_quant/test_acc": metrics_quant.get('test_acc', float('nan')),
            f"{name}_quant/test_precision": metrics_quant.get('test_precision', float('nan')),
            f"{name}_quant/test_recall": metrics_quant.get('test_recall', float('nan')),
            f"{name}_quant/test_f1": metrics_quant.get('test_f1', float('nan')),
            f"{name}_quant/size_mb": size_quant,
            f"{name}_quant/latency_ms": latency_quant
        })

        table_data.append([
            name, label,
            size_orig, size_quant,
            latency_orig, latency_quant,
            metrics_orig.get('test_loss', float('nan')),
            metrics_orig.get('test_acc', float('nan')),
            metrics_orig.get('test_precision', float('nan')),
            metrics_orig.get('test_recall', float('nan')),
            metrics_orig.get('test_f1', float('nan')),
            metrics_quant.get('test_loss', float('nan')),
            metrics_quant.get('test_acc', float('nan')),
            metrics_quant.get('test_precision', float('nan')),
            metrics_quant.get('test_recall', float('nan')),
            metrics_quant.get('test_f1', float('nan'))
        ])

    columns = [
        "Modelo", "Porcentaje Etiquetado",
        "Tamaño Original (MB)", "Tamaño Cuantizado Estático (MB)",
        "Latencia Original (ms)", "Latencia Cuantizada Estática (ms)",
        "Loss Original", "Acc Original", "Precision Original", "Recall Original", "F1 Original",
        "Loss Cuantizado Estático", "Acc Cuantizado Estático", "Precision Cuantizado Estático",
        "Recall Cuantizado Estático", "F1 Cuantizado Estático"
    ]
    table = wandb.Table(data=table_data, columns=columns)
    wandb_logger.experiment.log({"Model_Comparison": table})
    wandb_logger.experiment.finish()

    print("\n=== Resumen===")
    for row in table_data:
        print(f"\nModelo: {row[0]} ({row[1]}% etiquetado)")
        print(f"Tamaño: Original={row[2]:.2f} MB, Cuantizado={row[3]:.2f} MB")
        print(f"Latencia: Original={row[4]:.2f} ms, Cuantizado={row[5]:.2f} ms")
        print(f"Métricas Originales: Loss={row[6]:.4f}, Acc={row[7]:.4f}, Precision={row[8]:.4f}, Recall={row[9]:.4f}, F1={row[10]:.4f}")
        print(f"Métricas Cuantizadas: Loss={row[11]:.4f}, Acc={row[12]:.4f}, Precision={row[13]:.4f}, Recall={row[14]:.4f}, F1={row[15]:.4f}")

if __name__ == "__main__":
    quantize_and_compare()
