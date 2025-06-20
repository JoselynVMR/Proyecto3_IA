import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score
from autoencoder_unet import UNetAutoencoder

class ButterflyClassifier(pl.LightningModule):
    def __init__(self, encoder_weights_path=None, freeze_encoder=False, num_classes=30, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Cargar el autoencoder para extraer el encoder
        autoencoder = UNetAutoencoder()
        if encoder_weights_path:
            # Cargar pesos preentrenados del encoder
            state_dict = {
                'encoder1': torch.load(encoder_weights_path['encoder1']),
                'encoder2': torch.load(encoder_weights_path['encoder2']),
                'encoder3': torch.load(encoder_weights_path['encoder3']),
                'bottleneck': torch.load(encoder_weights_path['bottleneck'])
            }
            autoencoder.encoder1.load_state_dict(state_dict['encoder1'])
            autoencoder.encoder2.load_state_dict(state_dict['encoder2'])
            autoencoder.encoder3.load_state_dict(state_dict['encoder3'])
            autoencoder.bottleneck.load_state_dict(state_dict['bottleneck'])
    
        # Definir el encoder
        self.encoder = nn.Sequential(
            autoencoder.encoder1,
            autoencoder.pool1,
            autoencoder.encoder2,
            autoencoder.pool2,
            autoencoder.encoder3,
            autoencoder.pool3,
            autoencoder.bottleneck
        )

        # Congelar pesos del encoder si es necesario (para B1)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Capa fully connected para clasificación
        input_height = 128  # Dimensiones de entrada de las imágenes
        input_width = 128
        num_features = 512 * (input_height // 8) * (input_width // 8)  # 512 * 16 * 16
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        # Métricas
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.encoder(x)
        return self.fc(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = self.train_accuracy(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = self.val_accuracy(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = self.test_accuracy(logits, y)
        precision = self.test_precision(logits, y)
        recall = self.test_recall(logits, y)
        f1 = self.test_f1(logits, y)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)
        self.log("test_precision", precision, on_epoch=True)
        self.log("test_recall", recall, on_epoch=True)
        self.log("test_f1", f1, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)