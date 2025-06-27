import os
import torch
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
from torchvision.utils import make_grid
import gc

def extract_latents(model, dataloader, device):
    model.eval()
    latents_list = []

    # Extraer los vectores latentes
    print("Extrayendo vectores latentes...")
    with torch.no_grad(): 
        for batch_idx, batch in enumerate(dataloader):
            x_noisy, _ = batch 
            x = x_noisy.to(device)

            if model.use_variational:
                _, mu, _ = model.forward(x) 
                latent_representation = mu
            else:
                _, latent_rep_dae = model.forward(x)
                latent_representation = latent_rep_dae

            latents_list.append(latent_representation.cpu())

            # Limpieza de memoria
            del x, latent_representation
            if model.use_variational:
                del mu
            if device.type == 'cuda':
                torch.cuda.empty_cache() 
            gc.collect()

    all_latents_np = torch.cat(latents_list, dim=0).numpy()
    print(f"Vectores latentes extraídos: {all_latents_np.shape}")
    return all_latents_np

def plot_tsne_kmeans(latents, n_clusters=10, save_path="tsne_kmeans.png", model_name="Autoencoder"):
    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)

    tsne = TSNE(n_components=2, perplexity=15, max_iter=2000, n_iter_without_progress=500, random_state=42)
    latents_2d = tsne.fit_transform(latents_scaled)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, max_iter=1000, algorithm="lloyd", random_state=42)
    labels = kmeans.fit_predict(latents_scaled)

    plt.figure(figsize=(10, 8))
    colors = cm.get_cmap('gist_ncar', n_clusters)
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap=colors, alpha=0.6)
    # Calcular y graficar centroides
    for i in range(n_clusters):
        cluster_points = latents_2d[labels == i]
        if len(cluster_points) > 0:
            center = cluster_points.mean(axis=0)
            plt.scatter(center[0], center[1], marker='X', s=200, color='black', edgecolor='white', linewidth=1.5)
            plt.text(center[0], center[1], f' {i}', color='black', fontsize=12, ha='center', va='bottom')
        plt.title(f"t-SNE con K-means ({model_name})")
    plt.title(f"t-SNE con K-means ({model_name})")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")

    # Añadir los cluster
    handles, _ = scatter.legend_elements()
    plt.legend(handles=handles, labels=[f"Clúster {i}" for i in range(n_clusters)], title="Clústeres")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_tsne_only(latents, save_path="tsne_only.png", model_name="Autoencoder"):
    tsne = TSNE(n_components=2, perplexity=15, max_iter=2000, n_iter_without_progress=500)
    latents_2d = tsne.fit_transform(latents)

    colors = [random.random() for _ in range(len(latents_2d))]
    plt.figure(figsize=(10, 8))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=colors, cmap="viridis", alpha=0.6)
    plt.title(f"t-SNE de espacio latente ({model_name})")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_tsne_only(latents, save_path="tsne_only.png", model_name="Autoencoder",
                   perplexity=15, random_seed=42, true_labels=None):
    # Escalar los latentes
    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)

    # Aplicar t-SNE
    tsne = TSNE(n_components=2, perplexity=15, max_iter=2000, n_iter_without_progress=500, random_state=42)
    latents_2d = tsne.fit_transform(latents_scaled)

    plt.figure(figsize=(10, 8))

    # Se eligen los colores y etiquetas
    if true_labels is not None:
        num_classes = len(set(true_labels))
        cmap = cm.get_cmap('gist_ncar', num_classes)
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=true_labels, cmap=cmap, alpha=0.6)
        handles, _ = scatter.legend_elements()
        plt.legend(handles=handles, labels=[f"Clase {i}" for i in sorted(set(true_labels))], title="Clases Reales")
    else:
        plt.scatter(latents_2d[:, 0], latents_2d[:, 1], color='steelblue', alpha=0.6)

    plt.title(f"t-SNE de espacio latente ({model_name})")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def show_cluster(dataloader, model, n_clusters=30, examples_per_cluster=10, output_dir="clusters",
                 precomputed_latents=None, precomputed_labels=None):

    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    # Extracción de latentes
    all_latents_np = None
    all_indices = []

    if precomputed_latents is None:
        print("Extrayendo vectores latentes...")
        temp_latents_list = []
        with torch.no_grad():
            for batch_idx, (x_noisy, _) in enumerate(dataloader):
                x = x_noisy.to(device)

                if model.use_variational:
                    _, mu, _ = model.forward(x) 
                    latent_representation = mu
                else:
                    _, latent_rep_dae = model.forward(x)
                    latent_representation = latent_rep_dae

                temp_latents_list.append(latent_representation.cpu())

                # Recopilar indices originales del dataset
                start_idx = batch_idx * dataloader.batch_size
                end_idx = min((batch_idx + 1) * dataloader.batch_size, len(dataloader.dataset))
                all_indices.extend(range(start_idx, end_idx))

                del x, latent_representation
                if model.use_variational:
                    del mu 
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

        all_latents_np = torch.cat(temp_latents_list).numpy()
        print(f"Vectores latentes extraídos: {all_latents_np.shape}")
    else:
        if isinstance(precomputed_latents, torch.Tensor):
            all_latents_np = precomputed_latents.numpy()
        else:
            all_latents_np = precomputed_latents
        all_indices = list(range(len(dataloader.dataset)))
        print("Usando vectores latentes precalculados.")

    # Manejo de K-Means
    labels_np = precomputed_labels
    if labels_np is None:
        print(f"Aplicando K-Means con {n_clusters} clústeres...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=1000, algorithm="lloyd")
        labels_np = kmeans.fit_predict(all_latents_np)
        print("K-Means completado.")
    else:
        print("Usando etiquetas de clúster precalculadas.")


    print("Generando visualizaciones de clústeres...")
    # Iterar sobre TODOS los clústeres
    for i in range(n_clusters):
        cluster_original_indices = [all_indices[j] for j, label in enumerate(labels_np) if label == i]
        
        if len(cluster_original_indices) == 0:
            print(f"El clúster {i} está vacío, saltando la visualización.")
            continue

        # Seleccionar ejemplos aleatorios
        selected_original_indices = random.sample(cluster_original_indices, min(examples_per_cluster, len(cluster_original_indices)))
        
        images_to_plot = []
        for idx in selected_original_indices:
            _, clean_img = dataloader.dataset[idx]
            images_to_plot.append(clean_img)
        
        # Asegurarse de que las imagenes estén en el formato correcto
        images = torch.stack(images_to_plot)
        nrow_grid = int(examples_per_cluster**0.5)
        if nrow_grid == 0: nrow_grid = 1
        
        grid = make_grid(images, nrow=nrow_grid, normalize=True)
        
        fig_width = 10
        fig_height = fig_width * grid.shape[1] / grid.shape[2]
        
        plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title(f"Clúster {i} (Ejemplos: {len(selected_original_indices)})")
        plt.axis("off")
        plt.savefig(f"{output_dir}/cluster_{i}.png", dpi=300)
        plt.close()

        del images, grid, images_to_plot
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("✅ Análisis de espacio latente y visualización de clústeres completados.")
    
    return all_latents_np, labels_np


