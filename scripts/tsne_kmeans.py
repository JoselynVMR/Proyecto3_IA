import os
import torch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
from torchvision.utils import make_grid
from matplotlib import pyplot as plt


def extract_latents(model, dataloader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in dataloader:
            x, _ = batch
            x = x.to(device)
            x = model.encoder1(x)
            x = model.encoder2(x)
            x = model.encoder3(x)
            x = model.bottleneck(x)
            features.append(x.view(x.size(0), -1).cpu())
    return torch.cat(features, dim=0).numpy()

def plot_tsne_kmeans(latents, n_clusters=10, save_path="tsne_kmeans.png"):
    tsne = TSNE(n_components=2, perplexity=30)
    latents_2d = tsne.fit_transform(latents)

    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(latents)

    plt.figure(figsize=(10, 8))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.title("t-SNE de vectores latentes con K-means")
    plt.savefig(save_path)
    plt.close()


def show_cluster(dataloader, model, n_clusters=30, examples_per_cluster=10, output_dir="clusters"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    imgs, latents = [], []

    with torch.no_grad():
        for batch in dataloader:
            x, _ = batch
            x = x.to(device)
            encoded = model.encoder1(x)
            encoded = model.encoder2(encoded)
            encoded = model.encoder3(encoded)
            encoded = model.bottleneck(encoded)
            z = encoded.view(encoded.size(0), -1)
            imgs.append(x.cpu())
            latents.append(z.cpu())

    imgs = torch.cat(imgs)
    latents = torch.cat(latents)

    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(latents)

    for i in range(min(n_clusters, 5)):
        cluster_indices = (labels == i).nonzero()[0]
        if len(cluster_indices) == 0:
            continue
        selected_indices = cluster_indices[:examples_per_cluster]
        images = imgs[selected_indices]
        grid = make_grid(images, nrow=5, normalize=True)
        plt.figure(figsize=(10, 5))
        plt.imshow(grid.permute(1, 2, 0))
        plt.title(f"Cluster {i}")
        plt.axis("off")
        plt.savefig(f"{output_dir}/cluster_{i}.png")
        plt.close()

