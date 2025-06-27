# analyze_latents.py
import os
import sys
import torch
import warnings
from hydra import main
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.models.variational_autoencoder_unet import UNetAutoencoder
from scripts.noisy_pair_data_module import DataModule
from scripts.latent_analysis_utils import extract_latents, plot_tsne_only, plot_tsne_kmeans, show_cluster

GlobalHydra.instance().clear()

@main(config_path='../../configuration', config_name='config', version_base=None)
def analyze_latents(cfg: DictConfig):
    # 🧠 Cargar modelo
    model = UNetAutoencoder.load_from_checkpoint(
        cfg.experiment.latent_dae.paths.checkpoint,
        use_variational=cfg.experiment.latent_dae.model.use_variational
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 📦 Cargar datos sin ruido
    data_module = DataModule(
        hparams=cfg.experiment.latent_dae.datamodule,
        data_dir=cfg.experiment.latent_dae.paths.data_dir,
        noise_amount=cfg.experiment.latent_dae.datamodule.noise_amount
    )
    data_module.setup("fit")
    dataloader = data_module.train_dataloader()

    # 🔍 Extraer espacio latente
    latents = extract_latents(model, dataloader, device)

    # 📁 Crear carpeta de salida
    os.makedirs(cfg.experiment.latent_dae.paths.output_dir, exist_ok=True)

    # 📈 Visualizaciones
    plot_tsne_only(
        latents,
        save_path=os.path.join(cfg.experiment.latent_dae.paths.output_dir, "tsne_only.png"),
        model_name="dae" if not cfg.experiment.latent_dae.model.use_variational else "vdae"
    )
    plot_tsne_kmeans(
        latents,
        n_clusters=cfg.experiment.latent_dae.analysis.n_clusters,
        save_path=os.path.join(cfg.experiment.latent_dae.paths.output_dir, "tsne_kmeans.png"),
        model_name="dae" if not cfg.experiment.latent_dae.model.use_variational else "vdae"
    )
    show_cluster(
        dataloader=dataloader,
        model=model,
        n_clusters=cfg.experiment.latent_dae.analysis.n_clusters,
        examples_per_cluster=cfg.experiment.latent_dae.analysis.examples_per_cluster,
        output_dir=cfg.experiment.latent_dae.paths.output_dir
    )

    print("✅ Análisis de espacio latente completado.")

if __name__ == "__main__":
    analyze_latents()
