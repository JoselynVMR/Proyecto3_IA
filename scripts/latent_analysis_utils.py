import os
import torch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset
import gc # Add this at the top

def extract_latents(model, dataloader, device):
    model.eval()
    latents_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader): # Add batch_idx
            x, _ = batch
            x = x.to(device)

            e1 = model.encoder1(x)
            p1 = model.pool1(e1)

            e2 = model.encoder2(p1)
            p2 = model.pool2(e2)

            e3 = model.encoder3(p2)
            p3 = model.pool3(e3)

            b = model.bottleneck(p3)
            z = b.view(b.size(0), -1).cpu() # Move to CPU as soon as possible
            latents_list.append(z)

            # Explicitly delete intermediate tensors and clear cache more frequently
            del x, e1, p1, e2, p2, e3, p3, b
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect() # Force garbage collection

    return torch.cat(latents_list, dim=0).numpy()


def plot_tsne_kmeans(latents, n_clusters=10, save_path="tsne_kmeans.png"):
    tsne = TSNE(n_components=2, perplexity=15, max_iter=2000, n_iter_without_progress=500)
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
    """
    Visualiza ejemplos de imágenes para cada clúster encontrado en el espacio latente.

    Args:
        dataloader (torch.utils.data.DataLoader): El DataLoader para el conjunto de datos
                                                  (preferiblemente el de entrenamiento sin ruido).
        model (pytorch_lightning.LightningModule): El modelo Autoencoder (U-Net).
        n_clusters (int): Número de clústeres a formar con K-means.
        examples_per_cluster (int): Número de imágenes a mostrar por cada clúster.
        output_dir (str): Directorio donde se guardarán las imágenes de los clústeres.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval() # Poner el modelo en modo evaluación
    device = next(model.parameters()).device # Obtener el dispositivo actual del modelo

    all_latents = []
    # all_indices almacenará los índices originales de las muestras en el dataset completo.
    # Esto es crucial para poder cargar las imágenes específicas más tarde sin mantenerlas todas en memoria.
    all_indices = []

    print("Extrayendo vectores latentes...")
    # Primero, extrae todos los vectores latentes para todo el dataset (sin cargar todas las imágenes)
    with torch.no_grad(): # Desactivar el cálculo de gradientes para ahorrar memoria
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)

            # --- Ruta del encoder para obtener el vector latente ---
            e1 = model.encoder1(x)
            p1 = model.pool1(e1)

            e2 = model.encoder2(p1)
            p2 = model.pool2(e2)

            e3 = model.encoder3(p2)
            p3 = model.pool3(e3)

            b = model.bottleneck(p3)
            # Aplanar el bottleneck para obtener el vector latente y moverlo a CPU
            z = b.view(b.size(0), -1).cpu()

            all_latents.append(z)

            # Calcular y almacenar los índices originales de las muestras en este batch
            # len(dataloader.dataset) devuelve el tamaño total del dataset que el dataloader está usando.
            # Esto asume que el dataloader no ha sido modificado para devolver índices directamente.
            start_idx = batch_idx * dataloader.batch_size
            end_idx = min((batch_idx + 1) * dataloader.batch_size, len(dataloader.dataset))
            all_indices.extend(range(start_idx, end_idx))

            # Liberar memoria de forma agresiva
            del x, e1, p1, e2, p2, e3, p3, b, z
            if device.type == 'cuda':
                torch.cuda.empty_cache() # Limpiar la caché de memoria de la GPU
            gc.collect() # Forzar la recolección de basura de Python

    # Concatenar todos los vectores latentes
    all_latents = torch.cat(all_latents)
    print(f"Vectores latentes extraídos: {all_latents.shape}")

    print(f"Aplicando K-Means con {n_clusters} clústeres...")
    # Aplicar K-Means a los vectores latentes para obtener las etiquetas de clúster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init=10 es una buena práctica
    labels = kmeans.fit_predict(all_latents)
    print("K-Means completado.")

    print("Generando visualizaciones de clústeres...")
    # Ahora, para cada clúster, cargar solo las imágenes necesarias y visualizarlas
    # Se visualizan hasta un máximo de 5 clústeres (como se indica en el proyecto "Compare 10 imágenes de al menos 5 clusters")
    for i in range(min(n_clusters, 5)):
        # Obtener los índices originales de las imágenes que pertenecen al clúster actual
        cluster_original_indices = [all_indices[j] for j, label in enumerate(labels) if label == i]

        if len(cluster_original_indices) == 0:
            print(f"El clúster {i} está vacío, saltando.")
            continue

        # Seleccionar una muestra aleatoria de imágenes de este clúster
        # (asegurándose de no seleccionar más de las disponibles)
        selected_original_indices = random.sample(cluster_original_indices, min(examples_per_cluster, len(cluster_original_indices)))

        # Crear un DataLoader temporal para cargar solo las imágenes seleccionadas de este clúster
        # dataloader.dataset es el Dataset original (por ejemplo, ImageFolder o Subset)
        # que el 'dataloader' principal está utilizando.
        temp_dataset = Subset(dataloader.dataset, selected_original_indices)
        # Es crucial usar num_workers=0 aquí para evitar problemas de memoria o sincronización
        # con los workers del dataloader principal, especialmente para cargas tan pequeñas y específicas.
        temp_dataloader = DataLoader(temp_dataset, batch_size=examples_per_cluster, shuffle=False, num_workers=0)

        images_to_plot = []
        for img_batch, _ in temp_dataloader: # Iterar sobre el dataloader temporal para obtener las imágenes
            images_to_plot.append(img_batch)
        images = torch.cat(images_to_plot) # Concatenar todas las imágenes para la cuadrícula

        # Crear la cuadrícula de imágenes y guardarla
        grid = make_grid(images, nrow=5, normalize=True) # normalize=True para escalar los píxeles a [0, 1] para visualización
        plt.figure(figsize=(10, 5))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy()) # Permutar y mover a numpy para matplotlib
        plt.title(f"Clúster {i}")
        plt.axis("off")
        plt.savefig(f"{output_dir}/cluster_{i}.png")
        plt.close() # Cerrar la figura para liberar memoria

        # Liberar memoria de las imágenes y el dataloader temporal
        del images, grid, temp_dataset, temp_dataloader, images_to_plot
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("✅ Análisis de espacio latente y visualización de clústeres completados.")
