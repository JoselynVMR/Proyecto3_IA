import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import Dict
from salt_pepper import salt_and_pepper_noise

class AddSaltPepper:
    def __call__(self, img):
        img_tensor = transforms.ToTensor()(img)
        noisy_img = salt_and_pepper_noise(img_tensor)
        return noisy_img

class DataModule(pl.LightningDataModule):
    def __init__(self, hparams: Dict, data_dir: str, use_noise=False):
        super().__init__()
        self.save_hyperparameters(hparams)  # Guarda los hiperparámetros
        self.data_dir = data_dir
        self.use_noise = use_noise

        self.transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform_noise = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            AddSaltPepper() if self.use_noise else transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    #Here we can donwload the data
    def prepare_data(self):
        pass

    #Here we can load the data
    def setup(self, stage: str):
        train_transform=self.transform_noise if self.use_noise else self.transform
        # Cargar el conjunto de training
        train_data = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'train'),
            transform=train_transform
        )
        total_size = len(train_data)

        # Generar índices y revolverlos
        indices = np.arange(total_size)
        np.random.seed(self.hparams.seed)
        np.random.shuffle(indices)

        # Calcular el número de muestras etiquetadas
        label_count = int(total_size * self.hparams.label_pct)

        # Dividir los índices en etiquetados y no etiquetados
        labeled_indices = indices[:label_count]
        unlabeled_indices = indices[label_count:]

        # Crear subconjuntos
        self.labeled_ds = Subset(train_data, labeled_indices)
        self.unlabeled_ds = Subset(train_data, unlabeled_indices)

        # Cargar conjuntos de test y val
        self.val_ds = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'valid'),
            transform=self.transform
        )
        self.test_ds = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'test'),
            transform=self.transform
        )

    def labeled_dataloader(self):
        return DataLoader(
            self.labeled_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            persistent_workers=True
        )

    def unlabeled_dataloader(self):
        return DataLoader(
            self.unlabeled_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            persistent_workers=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.unlabeled_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            persistent_workers=True
        )