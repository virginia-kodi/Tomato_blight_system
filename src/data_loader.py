import os
import tensorflow as tf
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def get_data_generators():
    """
    Creates and returns the data generators for Train, Validation, and Test sets.
    Applies data augmentation to the training set.
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=config.VALIDATION_SPLIT + config.TEST_SPLIT, # Combined for initial split
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2
    )

    # Note: tf.keras ImageDataGenerator validation_split only does a 2-way split. 
    # To get a 3-way split (Train/Val/Test), we do a workaround:
    # 70% Train, 30% Temp (Val+Test). Then we take the temp and split it manually, 
    # or use tf.data.Dataset for more precise splitting.
    # For simplicity, we will use tf.keras.utils.image_dataset_from_directory
    
    print("Loading data...")
    dataset = tf.keras.utils.image_dataset_from_directory(
        config.PROCESSED_DATA_DIR,
        labels='inferred',
        label_mode='binary',
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    # Normalize pixel values
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = dataset.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Calculate split sizes
    dataset_size = dataset.cardinality().numpy()
    train_size = int(0.70 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_ds = normalized_ds.take(train_size)
    remaining_ds = normalized_ds.skip(train_size)
    val_ds = remaining_ds.take(val_size)
    test_ds = remaining_ds.skip(val_size)

    # Define augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    class_names = dataset.class_names
    print(f"Classes found: {class_names}")

    split_info = {
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size
    }

    return train_ds, val_ds, test_ds, class_names, split_info

if __name__ == '__main__':
    train, val, test, classes, splits = get_data_generators()
    print(f"Split sizes (batches): {splits}")
    print("Data loading script tested successfully.")
