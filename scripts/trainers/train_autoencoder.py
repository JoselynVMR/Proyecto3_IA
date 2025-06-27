import os
import sys
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra import main

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.models.autoencoder_unet import UNetAutoencoder
from scripts.data_module import DataModule
from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()

@main(config_path='../../configuration', config_name='config', version_base=None)
def train_autoencoder(cfg: DictConfig):

    os.environ["WANDB_API_KEY"] = cfg.key.api_key

    wandb_logger = instantiate(cfg.experiment.train_autoencoder.wandb)

    data_module = DataModule(
        hparams=cfg.experiment.train_autoencoder.datamodule,
        data_dir=cfg.experiment.train_autoencoder.paths.data_dir
    )

    model = UNetAutoencoder(learning_rate=cfg.experiment.train_autoencoder.model.learning_rate)

    early_stopping = instantiate(cfg.experiment.train_autoencoder.callbacks.early_stopping)
    checkpoint = instantiate(cfg.experiment.train_autoencoder.callbacks.checkpoint)

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[early_stopping, checkpoint],
        **cfg.experiment.train_autoencoder.trainer
    )

    trainer.fit(model, datamodule=data_module)

    os.makedirs(cfg.experiment.train_autoencoder.paths.weights_dir, exist_ok=True)
    torch.save(model.encoder1.state_dict(), f"{cfg.experiment.train_autoencoder.paths.weights_dir}/encoder_stage1.pth")
    torch.save(model.encoder2.state_dict(), f"{cfg.experiment.train_autoencoder.paths.weights_dir}/encoder_stage2.pth")
    torch.save(model.encoder3.state_dict(), f"{cfg.experiment.train_autoencoder.paths.weights_dir}/encoder_stage3.pth")
    torch.save(model.bottleneck.state_dict(), f"{cfg.experiment.train_autoencoder.paths.weights_dir}/encoder_bottleneck.pth")

    print(f"✅ Entrenamiento finalizado y pesos guardados en {cfg.experiment.train_autoencoder.paths.weights_dir}")

if __name__ == '__main__':
    train_autoencoder()
