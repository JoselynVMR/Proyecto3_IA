import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class UNetAutoencoder(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=3, init_features=64, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        features = init_features

        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 4, features * 8, kernel_size=3, padding=1),
            nn.InstanceNorm2d(features * 8),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(features * 8, features * 8, kernel_size=3, padding=1),
            nn.InstanceNorm2d(features * 8),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.InstanceNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.upconv3(b)
        d3 = self.decoder3(torch.cat((u3, e3), dim=1))

        u2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat((u2, e2), dim=1))

        u1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat((u1, e1), dim=1))

        return torch.sigmoid(self.final_conv(d1))

    def step(self, batch, stage):
        x_input, _ = batch
        x_hat = self(x_input)
        loss = F.mse_loss(x_hat, x_input)
        self.log(f"{stage}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
