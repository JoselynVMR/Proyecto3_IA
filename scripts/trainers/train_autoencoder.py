import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Añadir ruta raíz al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importar modelo y datamodule
from scripts.models.autoencoder_unet import UNetAutoencoder
from scripts.data_module import DataModule

def train_autoencoder():
    # 🔧 Hiperparámetros
    hparams = {
        'batch_size': 64,
        'num_workers': 4,
        'seed': 42,
        'label_pct': 0.3,   # Cambiar a 0.1 para la corrida con 10%
        'learning_rate': 1e-3
    }

    # 🔁 Derivar sufijo de ejecución
    label_pct = int(hparams['label_pct'] * 100)  # 30 o 10
    run_name = f"UNetAE_clean_{label_pct}pct"

    # 📁 Directorios
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "../../data/species_selected"))
    checkpoint_dir = f"checkpoints/autoencoder"
    checkpoint_name = f"best-UNetAE-v{label_pct}pct"
    weights_dir = f"weights/au{label_pct}"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # 📊 W&B Logger
    os.environ["WANDB_API_KEY"] = "757af0e5727478d40e4a586ed9175f733ee00948"
    #os.environ["WANDB_API_KEY"] = "3e7282c2a62557882828c8d06b01ec4b8f7135a1" # Llave Joselyn
    wandb_logger = WandbLogger(project="butterfly-ae", name=run_name)

    # 📦 DataModule sin ruido (autoencoder clásico)
    data_module = DataModule(hparams=hparams, data_dir=data_dir, use_noise=False)

    # 🧠 Modelo
    model = UNetAutoencoder(learning_rate=hparams['learning_rate'])

    # ⏹ Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=0.0005,
        verbose=True,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename=checkpoint_name,
        save_top_k=1,
        mode="min"
    )

    # ⚙️ Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        logger=wandb_logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        num_sanity_val_steps=0,
        log_every_n_steps=10
    )

    # 🚀 Entrenamiento
    trainer.fit(model, datamodule=data_module)

    # 💾 Guardar pesos del encoder (para modelos B)
    torch.save(model.encoder1.state_dict(), f"{weights_dir}/encoder_stage1.pth")
    torch.save(model.encoder2.state_dict(), f"{weights_dir}/encoder_stage2.pth")
    torch.save(model.encoder3.state_dict(), f"{weights_dir}/encoder_stage3.pth")
    torch.save(model.bottleneck.state_dict(), f"{weights_dir}/encoder_bottleneck.pth")

    print(f"Entrenamiento finalizado y pesos guardados en {weights_dir}")

# 🔁 Ejecución directa
if __name__ == '__main__':
    train_autoencoder()
