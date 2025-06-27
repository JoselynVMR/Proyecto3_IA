# train_classifier.py
import os
import sys
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra import main

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.models.classifier import ButterflyClassifier
from scripts.data_module import DataModule
from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()

@main(config_path='../../configuration', config_name='config', version_base=None)
def train_classifier(cfg: DictConfig):

    os.environ["WANDB_API_KEY"] = cfg.key.api_key

    encoder_weights = {
        'encoder1': os.path.join(cfg.experiment.train_classifier.paths.encoder_dir, "encoder_stage1.pth"),
        'encoder2': os.path.join(cfg.experiment.train_classifier.paths.encoder_dir, "encoder_stage2.pth"),
        'encoder3': os.path.join(cfg.experiment.train_classifier.paths.encoder_dir, "encoder_stage3.pth"),
        'bottleneck': os.path.join(cfg.experiment.train_classifier.paths.encoder_dir, "encoder_bottleneck.pth"),
    }

    classifiers = cfg.experiment.train_classifier.classifiers

    for config in classifiers:
        run_name = f"{config.name}_{cfg.experiment.train_classifier.datamodule.label}pct"
        wandb_logger = instantiate(cfg.experiment.train_classifier.wandb, name=run_name)

        data_module = DataModule(
            hparams=cfg.experiment.train_classifier.datamodule,
            data_dir=cfg.experiment.train_classifier.paths.data_dir
        )

        model = ButterflyClassifier(
            encoder_weights_path=encoder_weights if config.encoder_weights else None,
            freeze_encoder=config.freeze_encoder,
            num_classes=cfg.experiment.train_classifier.model.num_classes,
            learning_rate=cfg.experiment.train_classifier.model.learning_rate
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[instantiate(cfg.experiment.train_classifier.callbacks.early_stopping), instantiate(cfg.experiment.train_classifier.callbacks.checkpoint, filename=f"best-{config.name}")],
            **cfg.experiment.train_classifier.trainer
        )

        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)

        torch.save(model.state_dict(), os.path.join(cfg.experiment.train_classifier.paths.checkpoint_dir, f"{config.name}_{cfg.experiment.train_classifier.datamodule.label}pct_final.pth"))
        wandb_logger.experiment.finish()

    print("\n✅ Entrenamiento de clasificadores finalizado.")

if __name__ == '__main__':
    train_classifier()
