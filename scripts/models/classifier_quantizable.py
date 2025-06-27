import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score

from scripts.models.autoencoder_unet import UNetAutoencoder

class QuantizableButterflyClassifier(pl.LightningModule):
    def __init__(
        self,
        encoder_weights_path=None,
        freeze_encoder=False,
        num_classes=30,
        learning_rate=1e-3
    ):
        super().__init__()
        self.save_hyperparameters()

        # Inicializar autoencoder
        autoencoder = UNetAutoencoder()

        if encoder_weights_path:
            autoencoder.encoder1.load_state_dict(torch.load(encoder_weights_path['encoder1']))
            autoencoder.encoder2.load_state_dict(torch.load(encoder_weights_path['encoder2']))
            autoencoder.encoder3.load_state_dict(torch.load(encoder_weights_path['encoder3']))
            autoencoder.bottleneck.load_state_dict(torch.load(encoder_weights_path['bottleneck']))

        # Construcción del encoder
        self.encoder = nn.Sequential(
            autoencoder.encoder1,
            autoencoder.pool1,
            autoencoder.encoder2,
            autoencoder.pool2,
            autoencoder.encoder3,
            autoencoder.pool3,
            autoencoder.bottleneck
        )

        # Congelar encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Cálculo de dimensiones
        input_size = 128 
        encoded_channels = 512
        encoded_dim = (input_size // 8)  
        num_features = encoded_channels * encoded_dim * encoded_dim

        # Capa fully connected
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, num_classes)
        )

        # Métricas
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        self.loss_fn = nn.CrossEntropyLoss()

        # Se agrega soporte para cuantización estática
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)        
        x = self.encoder(x)
        x = self.fc(x)
        x = self.dequant(x)       
        return x

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
        self.log_dict({
            "test_loss": loss,
            "test_acc": acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        }, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}