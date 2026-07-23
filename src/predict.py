import os
import tensorflow as tf
import numpy as np
import cv2
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_and_prep_image(image_path):
    """Loads and preprocesses a single image for inference."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    # Always ensure it is converted to RGB (even if grayscale/segmented)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize
    img_resized = cv2.resize(img_rgb, config.IMG_SIZE)
    img_normalized = img_resized / 255.0
    
    # Add batch dimension
    return np.expand_dims(img_normalized, axis=0)

def predict(image_path):
    """Predicts the class of a single image and outputs confidence."""
    model_path = os.path.join(config.MODELS_DIR, 'model_v1.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
        
    model = tf.keras.models.load_model(model_path)
    
    img_tensor = load_and_prep_image(image_path)
    
    prediction = model.predict(img_tensor, verbose=0)[0][0]
    
    # Using 0.5 threshold for sigmoid output (0: Early_Blight, 1: Healthy assumed from directory ordering)
    # Note: Keras image_dataset_from_directory sorts labels alphanumerically.
    # Early_Blight comes before Healthy -> Early_Blight=0, Healthy=1
    if prediction > 0.5:
        predicted_class = "Healthy"
        confidence = prediction
    else:
        predicted_class = "Early_Blight"
        confidence = 1.0 - prediction
        
    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        try:
            result = predict(sys.argv[1])
            print(f"Class: {result['class']}")
            print(f"Confidence: {result['confidence']:.4f}")
        except Exception as e:
            print(f"Error: {e}")
