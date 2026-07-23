import os
import shutil
import cv2
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def process_and_copy_images():
    """
    Reads images from the original dataset (color, grayscale, segmented),
    filters out unnecessary classes, maps them to Early_Blight or Healthy,
    and copies them to the processed dataset directory (Unified Dataset).
    Converts grayscale and segmented images to 3-channel RGB to match 
    the color images and transfer learning model requirements.
    """
    print("Starting dataset preparation...")
    
    # Ensure processed dataset directories exist
    for target_class in config.CLASS_MAP.values():
        class_dir = os.path.join(config.PROCESSED_DATA_DIR, target_class)
        os.makedirs(class_dir, exist_ok=True)
    
    representations = ["color", "grayscale", "segmented"]
    
    copied_count = 0
    for rep in representations:
        rep_dir = os.path.join(config.DATASET_DIR, rep)
        if not os.path.exists(rep_dir):
            print(f"Warning: Directory {rep_dir} does not exist.")
            continue
            
        for orig_class in config.TARGET_CLASSES:
            orig_class_dir = os.path.join(rep_dir, orig_class)
            if not os.path.exists(orig_class_dir):
                print(f"Warning: Directory {orig_class_dir} does not exist.")
                continue
                
            target_class = config.CLASS_MAP[orig_class]
            target_class_dir = os.path.join(config.PROCESSED_DATA_DIR, target_class)
            
            for img_name in os.listdir(orig_class_dir):
                img_path = os.path.join(orig_class_dir, img_name)
                # Ensure it's an image
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                # To prevent naming collisions across representations, append representation name
                new_img_name = f"{rep}_{img_name}"
                target_path = os.path.join(target_class_dir, new_img_name)
                
                # For grayscale or segmented images, we need to ensure they are 3-channel
                # if they aren't already.
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # If grayscale, convert to RGB
                if len(img.shape) == 2 or (len(img.shape) == 3 and rep == "grayscale"):
                    # cv2.imread usually reads as BGR even for grayscale images if IMREAD_GRAYSCALE is not passed.
                    # We will force 3-channel representation.
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Convert back to BGR for saving with cv2.imwrite
                    img_to_save = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(target_path, img_to_save)
                else:
                    # Color or segmented, just copy
                    shutil.copy2(img_path, target_path)
                
                copied_count += 1
                if copied_count % 1000 == 0:
                    print(f"Processed {copied_count} images...")
                    
    print(f"Dataset preparation complete. Total images processed: {copied_count}")

if __name__ == '__main__':
    process_and_copy_images()
