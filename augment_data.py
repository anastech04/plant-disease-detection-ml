import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
from tqdm import tqdm 


BASE_DIR = r"C:\Documents\Dataset\Original\TomatoPepper_Final_Split\train"
TARGET_COUNT = 3000 


datagen = ImageDataGenerator(
    rotation_range=40,      # Rotate image
    width_shift_range=0.2,  # Move left/right
    height_shift_range=0.2, # Move up/down
    shear_range=0.2,        # Shear/slant
    zoom_range=0.3,         # Zoom in/out
    horizontal_flip=True,   # Mirror image
    brightness_range=[0.7, 1.3], # Darken/Lighten
    fill_mode='nearest'
)

print(f" Starting Massive Data Augmentation...")
print(f" Target: {TARGET_COUNT} images per class")
print(f" Source: {BASE_DIR}")


classes = os.listdir(BASE_DIR)

for class_name in classes:
    class_path = os.path.join(BASE_DIR, class_name)
    
    if not os.path.isdir(class_path):
        continue

  
    images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(images)
    
    needed = TARGET_COUNT - current_count
    
    print(f"\nProcessing: {class_name}")
    print(f"   - Current: {current_count}")
    
    if needed <= 0:
        print("   -  Already has enough images. Skipping.")
        continue
    
    print(f"   -  Generating {needed} new augmented images...")


    generated_count = 0
    
   
    pbar = tqdm(total=needed, unit="img")
    
    while generated_count < needed:
        random_image_path = random.choice(images)
        
        try:
            img = load_img(random_image_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape) 

            for batch in datagen.flow(x, batch_size=1, 
                                      save_to_dir=class_path, 
                                      save_prefix='aug', 
                                      save_format='jpg'):
                
                generated_count += 1
                pbar.update(1)
                break 
                
        except Exception as e:
            print(f"Error on image {random_image_path}: {e}")

    pbar.close()
    print(f"   -  Class {class_name} completed!")

print("\n Finished Data Augmentation for all classes.")