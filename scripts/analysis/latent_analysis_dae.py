import os
import sys
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Añadir ruta base
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Imports
from scripts.latent_analysis_utils import (
    extract_latents,
    plot_tsne_kmeans,
    show_cluster,
    plot_tsne_only
)
from scripts.models.autoencoder_unet import UNetAutoencoder
from scripts.data_module import DataModule

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "../../checkpoints/dae/best-DAE.ckpt")
    data_dir = os.path.abspath(os.path.join(current_dir, "../../data/species_selected"))

    # ⚙️ Modelo
    model = UNetAutoencoder.load_from_checkpoint(checkpoint_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ⚙️ DataModule sin ruido para extracción limpia
    hparams = {
        'batch_size': 16,
        'num_workers': 4,
        'seed': 42,
        'label_pct': 0.3  # No importa para DAE en este punto
    }

    data_module = DataModule(hparams=hparams, data_dir=data_dir, use_noise=False)
    data_module.setup("fit")
    dataloader = data_module.train_dataloader()

    # 🔍 Extracción y análisis
    latents = extract_latents(model, dataloader, device)
    plot_tsne_only(latents, save_path="tsne_only.png")
    plot_tsne_kmeans(latents, n_clusters=30, save_path="tsne_kmeans.png")
    show_cluster(
        dataloader=dataloader,
        model=model,
        n_clusters=30,
        examples_per_cluster=10,
        output_dir="clusters"
    )
    print("✅ Análisis de espacio latente completado.")
