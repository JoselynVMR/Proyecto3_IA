import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tsne_kmeans import extract_latents, plot_tsne_kmeans, show_cluster
from denoising_autoencoder import UNetAutoencoder
from dataModule import DataModule

model = UNetAutoencoder.load_from_checkpoint("checkpoints/dae/best-DAE.ckpt")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available () else "cpu")
model.to(device)

hparams = {
    'batch_size': 64,
    'num_workers': 4,
    'seed': 42,
    'label_pct': 0.1
}

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
data_dir = os.path.abspath(os.path.join(current_dir, "../../data/species_selected"))

# Usar datos sin ruido para análisis latente
data_module_no_noise = DataModule(hparams=hparams, data_dir=data_dir)
data_module_no_noise.setup("fit")
dataloader = data_module_no_noise.train_dataloader()

latents = extract_latents(model, dataloader, device)
plot_tsne_kmeans(latents, n_clusters=30, save_path="tsne_kmeans.png")
show_cluster(
    dataloader=dataloader,
    model=model,
    n_clusters=30,
    examples_per_cluster=10,
    output_dir="clusters"
)
