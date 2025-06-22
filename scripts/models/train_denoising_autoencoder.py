import os
import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Importa tu autoencoder y datamodule personalizado
from denoising_autoencoder import UNetAutoencoder
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataModule import DataModule

def train_denoising_autoencoder():
    # Hiperparámetros
    hparams = {
        'batch_size': 64,
        'num_workers': 4,
        'seed': 42,
        'label_pct': 0.1,  # Pequeño subset etiquetado
        'learning_rate': 1e-3
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "../../data/species_selected"))

    # Inicializar W&B
    os.environ["WANDB_API_KEY"] = "757af0e5727478d40e4a586ed9175f733ee00948"
    wandb_logger = WandbLogger(project="butterfly-dae", name="DAE_saltpepper_10pct")

    # DataModule con ruido Salt and Pepper en entrenamiento
    data_module = DataModule(hparams=hparams, data_dir=data_dir, use_noise=True)

    # Modelo
    model = UNetAutoencoder(learning_rate=hparams['learning_rate'])

    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/dae",
        filename="best-DAE",
        save_top_k=1,
        mode="min"
    )

    # Entrenador
    trainer = pl.Trainer(
        max_epochs=10,
        logger=wandb_logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        num_sanity_val_steps=0,
        log_every_n_steps=10
    )

    # Entrenamiento
    trainer.fit(model, datamodule=data_module)

    # Guardar pesos del encoder
    os.makedirs("weights", exist_ok=True)
    torch.save(model.encoder1.state_dict(), "weights/dae_encoder1.pth")
    torch.save(model.encoder2.state_dict(), "weights/dae_encoder2.pth")
    torch.save(model.encoder3.state_dict(), "weights/dae_encoder3.pth")
    torch.save(model.bottleneck.state_dict(), "weights/dae_bottleneck.pth")
    print("Pesos del encoder guardados exitosamente.")

# Para ejecución desde notebook si se importa:
if __name__ == '__main__':
    train_denoising_autoencoder()
