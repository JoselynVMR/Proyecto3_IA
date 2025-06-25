import os
import sys
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 📁 Añadir ruta base del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 🔁 Imports del proyecto
from scripts.models.variational_autoencoder_unet import UNetAutoencoder  # Mismo modelo pero sin uso de variacional
from scripts.noisy_pair_data_module import DataModule  # Usa el mismo DataModule coherente
from scripts.latent_analysis_utils import extract_latents, plot_tsne_only, plot_tsne_kmeans, show_cluster

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 🧠 Cargar el modelo DAE entrenado (no variacional)
    checkpoint_path = os.path.join(current_dir, "../../checkpoints/dae/best-DAE.ckpt")
    model = UNetAutoencoder.load_from_checkpoint(checkpoint_path, use_variational=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 📦 Configuración del DataModule sin ruido
    data_dir = os.path.abspath(os.path.join(current_dir, "../../data/species_selected"))
    hparams = {
        'batch_size': 64,
        'num_workers': 4,
        'seed': 42
    }

    data_module = DataModule(
        hparams=hparams,
        data_dir=data_dir,
        noise_amount=0.0  # Análisis sin ruido
    )
    data_module.setup("fit")
    dataloader = data_module.train_dataloader()

    # 🔍 Extracción y análisis del espacio latente
    latents = extract_latents(model, dataloader, device)

    # 📈 Visualizaciones
    output_dir = "./clusters/dae"
    os.makedirs(output_dir, exist_ok=True)

    plot_tsne_only(latents, save_path=os.path.join(output_dir, "tsne_only.png"), model_name="dae")
    plot_tsne_kmeans(latents, n_clusters=30, save_path=os.path.join(output_dir, "tsne_kmeans.png"), model_name="dae")
    show_cluster(
        dataloader=dataloader,
        model=model,
        n_clusters=30,
        examples_per_cluster=10,
        output_dir=output_dir
    )

    print("✅ Análisis de espacio latente para DAE completado.")
