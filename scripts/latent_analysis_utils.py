import os
import torch
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset
import gc # Add this at the top

def extract_latents(model, dataloader, device):
    model.eval() # Poner el modelo en modo de evaluación
    latents_list = []
    
    print("Extrayendo vectores latentes...")
    with torch.no_grad(): # Desactivar el cálculo de gradientes
        for batch_idx, batch in enumerate(dataloader):
            x_noisy, _ = batch # Usamos solo la imagen ruidosa para la entrada
            x = x_noisy.to(device)

            if model.use_variational:
                # Para VAE, el forward devuelve x_hat, mu, logvar. Nos interesa mu.
                _, mu, _ = model.forward(x) 
                latent_representation = mu
            else:
                # Para DAE, el forward devuelve x_hat, latent_representation. Nos interesa latent_representation.
                _, latent_rep_dae = model.forward(x)
                latent_representation = latent_rep_dae

            latents_list.append(latent_representation.cpu()) # Mover a CPU y añadir a la lista

            # Limpieza de memoria
            del x, latent_representation
            if model.use_variational:
                del mu # Limpiar mu solo si fue creada
            if device.type == 'cuda':
                torch.cuda.empty_cache() # Vaciar caché de CUDA
            gc.collect() # Ejecutar recolector de basura

    all_latents_np = torch.cat(latents_list, dim=0).numpy() # Concatenar y convertir a NumPy
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
    # Calcular y graficar centroides (aproximados en el espacio 2D)
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

    # 5. Añadir una leyenda para los clústeres
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
    # 1. Escalar los latentes
    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)

    # 2. Aplicar t-SNE con random_state para reproducibilidad
    tsne = TSNE(n_components=2, perplexity=15, max_iter=2000, n_iter_without_progress=500, random_state=42)
    latents_2d = tsne.fit_transform(latents_scaled)

    plt.figure(figsize=(10, 8))

    if true_labels is not None:
        # Usar las etiquetas verdaderas para colorear
        num_classes = len(set(true_labels))
        # Elegir un colormap adecuado para el número de clases
        # 'gist_ncar' o 'hsv' son buenos para muchos colores distintos
        cmap = cm.get_cmap('gist_ncar', num_classes)
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=true_labels, cmap=cmap, alpha=0.6)
        # Añadir leyenda si hay true_labels
        handles, _ = scatter.legend_elements()
        plt.legend(handles=handles, labels=[f"Clase {i}" for i in sorted(set(true_labels))], title="Clases Reales")
    else:
        # Si no hay etiquetas verdaderas, usar un color único (los colores aleatorios no son útiles)
        plt.scatter(latents_2d[:, 0], latents_2d[:, 1], color='steelblue', alpha=0.6)

    plt.title(f"t-SNE de espacio latente ({model_name})")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300) # Guardar con mayor resolución
    plt.close()

def show_cluster(dataloader, model, n_clusters=30, examples_per_cluster=10, output_dir="clusters",
                 precomputed_latents=None, precomputed_labels=None):

    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    # --- Manejo de la extracción de latentes ---
    all_latents_np = None # Para almacenar los latentes como array de NumPy
    all_indices = []

    if precomputed_latents is None:
        print("Extrayendo vectores latentes...")
        temp_latents_list = []
        with torch.no_grad():
            for batch_idx, (x_noisy, _) in enumerate(dataloader):
                x = x_noisy.to(device)

                if model.use_variational:
                    # Para VAE, el forward devuelve x_hat, mu, logvar. Nos interesa mu.
                    _, mu, _ = model.forward(x) 
                    latent_representation = mu
                else:
                    # Para DAE, el forward devuelve x_hat, latent_representation. Nos interesa latent_representation.
                    _, latent_rep_dae = model.forward(x)
                    latent_representation = latent_rep_dae

                temp_latents_list.append(latent_representation.cpu())

                # Recopilar índices originales del dataset
                start_idx = batch_idx * dataloader.batch_size
                end_idx = min((batch_idx + 1) * dataloader.batch_size, len(dataloader.dataset))
                all_indices.extend(range(start_idx, end_idx))

                del x, latent_representation # Limpiar variables
                if model.use_variational:
                    del mu # Liberar mu solo si fue creada
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

        all_latents_np = torch.cat(temp_latents_list).numpy()
        print(f"Vectores latentes extraídos: {all_latents_np.shape}")
    else:
        # Usar latentes precalculados
        if isinstance(precomputed_latents, torch.Tensor):
            all_latents_np = precomputed_latents.numpy()
        else:
            all_latents_np = precomputed_latents # Ya es NumPy
        # Si se proporcionan latentes precomputados, asumimos que corresponden al dataset completo.
        # En un escenario real, deberías asegurarte de que estos índices también coincidan.
        all_indices = list(range(len(dataloader.dataset)))
        print("Usando vectores latentes precalculados.")

    # --- Manejo de K-Means ---
    labels_np = precomputed_labels
    if labels_np is None:
        print(f"Aplicando K-Means con {n_clusters} clústeres...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=1000, algorithm="lloyd")
        labels_np = kmeans.fit_predict(all_latents_np)
        print("K-Means completado.")
    else:
        print("Usando etiquetas de clúster precalculadas.")


    print("Generando visualizaciones de clústeres...")
    # Iterar sobre TODOS los clústeres (de 0 a n_clusters-1)
    for i in range(n_clusters):
        cluster_original_indices = [all_indices[j] for j, label in enumerate(labels_np) if label == i]
        
        if len(cluster_original_indices) == 0:
            print(f"El clúster {i} está vacío, saltando la visualización.")
            continue

        # Seleccionar ejemplos aleatorios
        selected_original_indices = random.sample(cluster_original_indices, min(examples_per_cluster, len(cluster_original_indices)))
        
        images_to_plot = []
        for idx in selected_original_indices:
            # Asumiendo que dataloader.dataset[idx] devuelve (imagen_ruidosa, imagen_limpia)
            # Y que queremos la imagen limpia (x_clean) para visualizar el contenido del clúster.
            _, clean_img = dataloader.dataset[idx]
            images_to_plot.append(clean_img)
        
        # Asegurarse de que las imágenes estén en el formato correcto para make_grid (Tensor)
        images = torch.stack(images_to_plot)

        # Ajustar nrow para que la cuadrícula sea más cuadrada, si es posible
        nrow_grid = int(examples_per_cluster**0.5)
        if nrow_grid == 0: nrow_grid = 1 # Evitar división por cero si examples_per_cluster es 0
        
        grid = make_grid(images, nrow=nrow_grid, normalize=True)
        
        # Ajustar el tamaño de la figura dinámicamente
        fig_width = 10
        fig_height = fig_width * grid.shape[1] / grid.shape[2] # Mantiene la proporción de aspecto
        
        plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title(f"Clúster {i} (Ejemplos: {len(selected_original_indices)})")
        plt.axis("off")
        plt.savefig(f"{output_dir}/cluster_{i}.png", dpi=300) # Guardar con alta resolución
        plt.close()

        del images, grid, images_to_plot
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("✅ Análisis de espacio latente y visualización de clústeres completados.")
    
    return all_latents_np, labels_np


