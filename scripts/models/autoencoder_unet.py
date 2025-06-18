import torch
import torch.nn as nn
import pytorch_lightning as pl

class UNetAutoencoder(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate

        # Encoder
        self.encoder1 = self.conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.loss_fn = nn.MSELoss()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        u3 = self.upconv3(b)
        d3 = self.decoder3(torch.cat([u3, e3], dim=1))

        u2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat([u2, e2], dim=1))

        u1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat([u1, e1], dim=1))

        return torch.sigmoid(self.final_conv(d1))

    def training_step(self, batch, batch_idx):
        x, _ = batch  # Ignoramos etiquetas
        x_hat = self.forward(x)
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_fn(x_hat, x)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
