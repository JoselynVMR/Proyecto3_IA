import os
import torch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
from torchvision.utils import make_grid


def extract_latents(model, dataloader, device):
    model.eval()
    latents_list = []

    with torch.no_grad():
        for batch in dataloader:
            x, _ = batch
            x = x.to(device)

            e1 = model.encoder1(x)
            p1 = model.pool1(e1)

            e2 = model.encoder2(p1)
            p2 = model.pool2(e2)

            e3 = model.encoder3(p2)
            p3 = model.pool3(e3)

            b = model.bottleneck(p3)
            z = b.view(b.size(0), -1).cpu()
            latents_list.append(z)

            del x, e1, e2, e3, p1, p2, p3, b, z
            torch.cuda.empty_cache()

    return torch.cat(latents_list, dim=0).numpy()


def plot_tsne_kmeans(latents, n_clusters=10, save_path="tsne_kmeans.png"):
    tsne = TSNE(n_components=2, perplexity=20, max_iter=2000, n_iter_without_progress=500)
    latents_2d = tsne.fit_transform(latents)

    kmeans = KMeans(n_clusters=n_clusters, max_iter=500, algorithm="lloyd")
    labels = kmeans.fit_predict(latents)

    plt.figure(figsize=(10, 8))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.title("t-SNE de vectores latentes con K-means")
    plt.savefig(save_path)
    plt.close()


def plot_tsne_only(latents, save_path="tsne_only.png"):
    tsne = TSNE(n_components=2, perplexity=15, max_iter=2000, n_iter_without_progress=500)
    latents_2d = tsne.fit_transform(latents)

    colors = [random.random() for _ in range(len(latents_2d))]
    plt.figure(figsize=(10, 8))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=colors, cmap="viridis", alpha=0.6)
    plt.title("Visualización del espacio latente con t-SNE")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.tight_layout()
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
