import os
import sys
import tensorflow as tf

# Ensure sibling modules in src/ and parent config are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_data_generators
from model import build_model
import logger
import config



def train_model():
    """Trains the CNN model and logs curves."""
    print("Initializing Data Loaders...")
    train_ds, val_ds, _, _, split_info = get_data_generators()
    
    print("Building Model...")
    model = build_model()
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config.MODELS_DIR, 'model_v1.h5'),
            save_best_only=True,
            monitor='val_loss'
        )
    ]
    
    print("Starting Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks
    )
    
    print("Training Completed. Saving history and curves...")
    logger.save_training_history(history)
    logger.save_training_curves(history)
    logger.save_dataset_split_chart(
        split_info['train_size'],
        split_info['val_size'],
        split_info['test_size']
    )

if __name__ == '__main__':
    train_model()
