import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class UNetAutoencoder(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=3, init_features=64, latent_dim=256, learning_rate=1e-3, use_variational=False):
        super().__init__()
        self.save_hyperparameters()
        self.use_variational = use_variational
        self.latent_dim = latent_dim

        features = init_features

        # Encoder
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.conv_bottleneck = nn.Conv2d(features * 4, features * 8, kernel_size=3, padding=1)
        self.bottleneck_spatial_size = 16 
        self.bottleneck_channels = features * 8 # 64 * 8 = 512
        self.bottleneck_flat_size = self.bottleneck_channels * self.bottleneck_spatial_size * self.bottleneck_spatial_size # 512 * 16 * 16 = 131072

        if self.use_variational:
            self.fc_mu = nn.Linear(self.bottleneck_flat_size, latent_dim)
            self.fc_logvar = nn.Linear(self.bottleneck_flat_size, latent_dim)
            self.fc_decode = nn.Linear(latent_dim, self.bottleneck_flat_size)
        else: # Para el DAE, añadir capas para una representación latente compacta
            self.fc_dae_latent = nn.Linear(self.bottleneck_flat_size, latent_dim)
            self.fc_dae_decode = nn.Linear(latent_dim, self.bottleneck_flat_size)

        self.decode_conv = nn.Sequential(
            nn.BatchNorm2d(features * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(features * 8, features * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(inplace=True)
        )

        # Decoder
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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        b = self.conv_bottleneck(p3)
        return e1, e2, e3, b

    def decode(self, b, e1, e2, e3):
        u3 = self.upconv3(b)
        d3 = self.decoder3(torch.cat((u3, e3), dim=1))
        u2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat((u2, e2), dim=1))
        u1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat((u1, e1), dim=1))
        return torch.sigmoid(self.final_conv(d1))

    def forward(self, x):
        e1, e2, e3, b = self.encode(x)
        b_flat = b.view(b.size(0), -1)

        if self.use_variational:
            mu = self.fc_mu(b_flat)
            logvar = self.fc_logvar(b_flat)
            z = self.reparameterize(mu, logvar)
            b_decoded_flat = self.fc_decode(z)
            b_decoded_conv_input = b_decoded_flat.view(b.size()) 
            b_processed = self.decode_conv(b_decoded_conv_input)
            x_hat = self.decode(b_processed, e1, e2, e3)
            return x_hat, mu, logvar
        else:
            latent_representation = self.fc_dae_latent(b_flat)
            b_reconstructed_flat = self.fc_dae_decode(latent_representation)
            b_reconstructed_conv_input = b_reconstructed_flat.view(b.size()) 
            b_processed = self.decode_conv(b_reconstructed_conv_input)
            x_hat = self.decode(b_processed, e1, e2, e3)
            return x_hat, latent_representation

    def step(self, batch, stage):
        x_noisy, x_clean = batch

        if self.use_variational:
            x_hat, mu, logvar = self.forward(x_noisy) # Desempaqueta correctamente para VAE
            recon_loss = F.mse_loss(x_hat, x_clean, reduction="mean")
            kl_div_weight = getattr(self.hparams, 'kl_weight', 1e-4)
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_div * kl_div_weight
            self.log_dict({f"{stage}_loss": loss, f"{stage}_recon": recon_loss, f"{stage}_kl": kl_div})
            return loss
        else:
            # LÍNEA CORREGIDA: Desempaquetar la tupla para obtener x_hat y descartar la representación latente
            x_hat, _ = self.forward(x_noisy) 
            loss = F.mse_loss(x_hat, x_clean, reduction="mean")
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
