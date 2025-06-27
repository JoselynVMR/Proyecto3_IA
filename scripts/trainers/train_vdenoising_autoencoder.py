import os
import sys
import torch
import pytorch_lightning as pl
from hydra import main
from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.models.variational_autoencoder_unet import UNetAutoencoder
from scripts.noisy_pair_data_module import DataModule

GlobalHydra.instance().clear()

@main(config_path='../../configuration', config_name='config', version_base=None)
def train_vdae(cfg: DictConfig):

    os.environ["WANDB_API_KEY"] = cfg.key.api_key
    os.environ["LOKY_MAX_CPU_COUNT"] = "6"

    wandb_logger = instantiate(cfg.experiment.train_vdae.wandb)

    data_module = DataModule(
        hparams=cfg.experiment.train_vdae.datamodule,
        data_dir=cfg.experiment.train_vdae.paths.data_dir,
        noise_amount=cfg.experiment.train_vdae.datamodule.noise_amount,
        image_size=cfg.experiment.train_vdae.datamodule.image_size
    )

    model = UNetAutoencoder(
        latent_dim=cfg.experiment.train_vdae.model.latent_dim,
        learning_rate=cfg.experiment.train_vdae.model.learning_rate,
        use_variational=cfg.experiment.train_vdae.model.use_variational
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[
            instantiate(cfg.experiment.train_vdae.callbacks.early_stopping),
            instantiate(cfg.experiment.train_vdae.callbacks.checkpoint)
        ],
        **cfg.experiment.train_vdae.trainer
    )

    trainer.fit(model, datamodule=data_module)

    # Guardar pesos
    torch.save(model.encoder1.state_dict(), os.path.join(cfg.experiment.train_vdae.paths.weights_dir, "vdae_encoder1.pth"))
    torch.save(model.encoder2.state_dict(), os.path.join(cfg.experiment.train_vdae.paths.weights_dir, "vdae_encoder2.pth"))
    torch.save(model.encoder3.state_dict(), os.path.join(cfg.experiment.train_vdae.paths.weights_dir, "vdae_encoder3.pth"))
    torch.save(model.conv_bottleneck.state_dict(), os.path.join(cfg.experiment.train_vdae.paths.weights_dir, "vdae_bottleneck.pth"))

    print("✅ Pesos del encoder guardados exitosamente.")

if __name__ == '__main__':
    train_vdae()
