# train_classifier.py
import os
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from classifier import ButterflyClassifier
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataModule import DataModule
import torch

if __name__ == '__main__':
    # Configuración de hiperparámetros y rutas
    hparams = {
        'batch_size': 64,
        'num_workers': 4,
        'seed': 42,
        'learning_rate': 1e-3,
        'label_pct': 0.1
    }

    # Directorio de datos
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    data_dir = os.path.abspath(os.path.join(current_dir, "../../data/species_selected"))

    # Pesos preentrenados del encoder
    encoder_weights = {
        'encoder1': "encoder_stage1.pth",
        'encoder2': "encoder_stage2.pth",
        'encoder3': "encoder_stage3.pth",
        'bottleneck': "encoder_bottleneck.pth"
    }

    # Configuraciones de los clasificadores
    classifiers = [
        {'name': 'Classifier_A', 'encoder_weights': None, 'freeze_encoder': False},
        {'name': 'Classifier_B1', 'encoder_weights': encoder_weights, 'freeze_encoder': True},
        {'name': 'Classifier_B2', 'encoder_weights': encoder_weights, 'freeze_encoder': False}
    ]

    # Inicializar WandB
    os.environ["WANDB_API_KEY"] = "757af0e5727478d40e4a586ed9175f733ee00948" # Llave Esteban
    #os.environ["WANDB_API_KEY"] = "3e7282c2a62557882828c8d06b01ec4b8f7135a1"  # Llave Joselyn
    for config in classifiers:
        wandb_logger = WandbLogger(
            project="butterfly-classifier",
            name=f"{config['name']}_{int(hparams['label_pct']*100)}pct",
            config=hparams
        )

        # Inicializar módulo de datos
        data_module = DataModule(hparams=hparams, data_dir=data_dir)

        # Inicializar modelo
        model = ButterflyClassifier(
            encoder_weights_path=config['encoder_weights'],
            freeze_encoder=config['freeze_encoder'],
            num_classes=30,
            learning_rate=hparams['learning_rate']
        )

        # Callbacks: EarlyStopping y Checkpoint
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=5,
            verbose=True,
            mode="min"
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints/classifier",
            filename=f"best-{config['name']}",
            save_top_k=1,
            save_weights_only=True,
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
        trainer.test(model, datamodule=data_module)

        # Guardar modelo
        torch.save(model.state_dict(), f"checkpoints/classifier/{config['name']}_{int(hparams['label_pct']*100)}pct_final.pth")

        wandb.finish()

    print("Entrenamiento de clasificadores completado.")