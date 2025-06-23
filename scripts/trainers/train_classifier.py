import os
import sys
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# 📁 Agregar ruta raíz
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 🧠 Importar clases del proyecto
from scripts.models.classifier import ButterflyClassifier
from scripts.data_module import DataModule

def main():
    # 🔧 Hiperparámetros
    hparams = {
        'batch_size': 64,
        'num_workers': 4,
        'seed': 42,
        'learning_rate': 1e-3,
        'label_pct': 0.3  # Cambiar a 0.3 según el experimento
    }

    label_pct = int(hparams['label_pct'] * 100)

    # 📁 Rutas base
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "../../data/species_selected"))
    weight_dir = os.path.abspath(os.path.join(current_dir, f"../../weights/au{label_pct}"))
    checkpoint_dir = os.path.abspath(os.path.join(current_dir, f"../../checkpoints/classifier/cs{label_pct}"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 🧱 Rutas de pesos del encoder preentrenado
    encoder_weights = {
        'encoder1': os.path.join(weight_dir, "encoder_stage1.pth"),
        'encoder2': os.path.join(weight_dir, "encoder_stage2.pth"),
        'encoder3': os.path.join(weight_dir, "encoder_stage3.pth"),
        'bottleneck': os.path.join(weight_dir, "encoder_bottleneck.pth"),
    }

    # 🧪 Configuración de clasificadores
    classifiers = [
        {'name': 'Classifier_A', 'encoder_weights': None, 'freeze_encoder': False},
        {'name': 'Classifier_B1', 'encoder_weights': encoder_weights, 'freeze_encoder': True},
        {'name': 'Classifier_B2', 'encoder_weights': encoder_weights, 'freeze_encoder': False}
    ]

    # 🔁 Entrenamiento de cada clasificador
    os.environ["WANDB_API_KEY"] = "757af0e5727478d40e4a586ed9175f733ee00948"
    #os.environ["WANDB_API_KEY"] = "3e7282c2a62557882828c8d06b01ec4b8f7135a1" # Llave Joselyn

    for config in classifiers:
        run_name = f"{config['name']}_{label_pct}pct"
        wandb_logger = WandbLogger(project="butterfly-classifier", name=run_name, config=hparams)

        # 📦 Inicializar DataModule
        data_module = DataModule(hparams=hparams, data_dir=data_dir)

        # 🧠 Inicializar modelo
        model = ButterflyClassifier(
            encoder_weights_path=config['encoder_weights'],
            freeze_encoder=config['freeze_encoder'],
            num_classes=30,
            learning_rate=hparams['learning_rate']
        )

        # ⏹ Callbacks
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=0.0005,
            verbose=True,
            mode="min"
        )
        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            dirpath=checkpoint_dir,
            filename=f"best-{config['name']}",
            save_top_k=1,
            save_weights_only=True,
            mode="min"
        )

        # ⚙️ Entrenador
        trainer = pl.Trainer(
            max_epochs=10,
            logger=wandb_logger,
            callbacks=[early_stop, checkpoint],
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            num_sanity_val_steps=0,
            log_every_n_steps=10
        )

        # 🚀 Entrenamiento y prueba
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)

        # 💾 Guardar pesos finales
        final_path = os.path.join(checkpoint_dir, f"{config['name']}_{label_pct}pct_final.pth")
        torch.save(model.state_dict(), final_path)

        wandb.finish()

    print("✅ Entrenamiento de clasificadores finalizado.")


if __name__ == '__main__':
    main()
