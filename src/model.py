import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def build_model():
    """
    Builds and compiles the CNN classification model using Transfer Learning (MobileNetV2).
    """
    input_shape = config.IMG_SIZE + (3,)
    
    # Load MobileNetV2 without the top dense layers
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base layers
    base_model.trainable = False
    
    # Build the classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    # Combine the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

if __name__ == '__main__':
    model = build_model()
    model.summary()
