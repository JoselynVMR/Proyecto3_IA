import os
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from autoencoder_unet import UNetAutoencoder
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataModuleE1 import DataModuleE1

import torch
import torch.nn.functional as F

if __name__ == '__main__':
    # Configuración de hiperparámetros y rutas
    hparams = {
        'batch_size': 64,
        'num_workers': 4,
        'seed': 42,
        'label_pct': 0.3,  # Ajustar a 0.1 para el segundo experimento
        'learning_rate': 1e-3
    }

    # Define el directorio de datos ya preprocesado (species_selected)
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    data_dir = os.path.abspath(os.path.join(current_dir, "../../data/species_selected"))

    print(data_dir)

    # Inicializar WandB
    os.environ["WANDB_API_KEY"] = "757af0e5727478d40e4a586ed9175f733ee00948" # Llave Esteban
    #os.environ["WANDB_API_KEY"] = "3e7282c2a62557882828c8d06b01ec4b8f7135a1" # Llave Joselyn
    wandb_logger = WandbLogger(project="butterfly-autoencoder", name="UNetAE_70pct")

    # Inicializar módulo de datos
    data_module = DataModuleE1(hparams=hparams, data_dir=data_dir)

    # Inicializar modelo
    model = UNetAutoencoder(learning_rate=hparams['learning_rate'])

    # Callbacks: EarlyStopping y Checkpoint
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/autoencoder",
        filename="best-UNetAE",
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
    torch.save(model.encoder1.state_dict(), "encoder_stage1.pth")
    torch.save(model.encoder2.state_dict(), "encoder_stage2.pth")
    torch.save(model.encoder3.state_dict(), "encoder_stage3.pth")
    torch.save(model.bottleneck.state_dict(), "encoder_bottleneck.pth")

    print("Pesos del encoder guardados exitosamente.")
