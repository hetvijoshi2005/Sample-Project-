import tensorflow as tf
import numpy as np
import os
import random

# --- 1. SET YOUR PATHS ---
DATASET_DIR = r"dataset" 
# This should be the path to your folder that contains "train" and "test"

# --- 2. SET NEW CONSTANTS ---
# We get this from our _3_model.py file
IMG_SHAPE = (128, 128, 3) 
# We'll set this here, but it will be auto-detected
NUM_CLASSES = 297 # (196 cars + 101 food)


def load_data_from_folders(data_dir):
    """
    Scans the data_dir/train/ folder to create the label-to-indices mapping.
    This is the new "loader" for our custom dataset.
    
    Returns:
    - (x_train, y_train): A list of filepaths and a list of integer labels
    - label_to_indices: The dict {0: [idx1, idx2...], 1: [idx3, ...]}
    - class_names: A list of the string names (e.g., "apple_pie")
    """
    print("Loading data from folders...")
    train_dir = os.path.join(data_dir, "train")
    
    # 1. Get all class names (folder names)
    class_names = sorted(os.listdir(train_dir))
    global NUM_CLASSES
    NUM_CLASSES = len(class_names)
    
    if NUM_CLASSES == 0:
        raise FileNotFoundError(f"No class folders found in {train_dir}.")
        
    print(f"Found {NUM_CLASSES} classes.")
    
    x_train_paths = [] # This will be a list of file paths
    y_train_labels = [] # This will be a list of integer labels (0, 1, 2...)
    label_to_indices = {i: [] for i in range(NUM_CLASSES)}
    
    current_image_index = 0
    
    # 2. Loop over each class folder
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(train_dir, class_name)
        
        # 3. Loop over every image in that class folder
        for image_name in os.listdir(class_dir):
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(class_dir, image_name)
                
                # Add the path to our list
                x_train_paths.append(image_path)
                
                # Add the integer label
                y_train_labels.append(label_idx)
                
                # Add the *index* of this image to our mapping dict
                label_to_indices[label_idx].append(current_image_index)
                
                current_image_index += 1
                
        print(f"  Processed class {label_idx+1}/{NUM_CLASSES}: {class_name}")

    print(f"\nTotal training images found: {len(x_train_paths)}")
    
    # NOTE: x_train_paths is NOT the image data, just the filepaths.
    # The actual image data will be loaded on-the-fly by the generator.
    return (x_train_paths, np.array(y_train_labels)), label_to_indices, class_names

def preprocess_image(image_path):
    """
    Loads and preprocesses a single image.
    This is our new preprocessing function for high-res images.
    """
    img = tf.io.read_file(image_path)
    # Decode as 3-channel (color)
    img = tf.image.decode_jpeg(img, channels=3) 
    # Resize to our model's expected shape
    img = tf.image.resize(img, [IMG_SHAPE[0], IMG_SHAPE[1]])
    # Normalize pixels from 0-255 to 0.0-1.0
    img = img / 255.0  
    return img

def get_triplet(label_to_indices, num_classes):
    """
    Generates a single (Anchor, Positive, Negative) triplet of *indices*.
    This logic is identical to our old script.
    """
    anchor_class_idx = np.random.randint(0, num_classes)
    negative_class_idx = (anchor_class_idx + np.random.randint(1, num_classes)) % num_classes
    
    anchor_idx, positive_idx = np.random.choice(
        label_to_indices[anchor_class_idx], 2, replace=False)
    
    negative_idx = np.random.choice(label_to_indices[negative_class_idx], 1)[0]
    
    return anchor_idx, positive_idx, negative_idx

def triplet_generator(batch_size, x_train_paths, label_to_indices, num_classes):
    """
    A generator that yields batches of (Anchor, Positive, Negative) images.
    This is now a tf.data.Dataset for high performance.
    """
    print(f"\nStarting triplet generator with batch size {batch_size}...")
    
    while True:
        anchor_images = np.zeros((batch_size, *IMG_SHAPE), dtype='float32')
        positive_images = np.zeros((batch_size, *IMG_SHAPE), dtype='float32')
        negative_images = np.zeros((batch_size, *IMG_SHAPE), dtype='float32')
        
        for i in range(batch_size):
            # 1. Get triplet indices
            anchor_idx, positive_idx, negative_idx = get_triplet(
                label_to_indices, num_classes
            )
            
            # 2. Get the filepaths for those indices
            anchor_path = x_train_paths[anchor_idx]
            positive_path = x_train_paths[positive_idx]
            negative_path = x_train_paths[negative_idx]
            
            # 3. Load and preprocess the actual images
            anchor_images[i] = preprocess_image(anchor_path)
            positive_images[i] = preprocess_image(positive_path)
            negative_images[i] = preprocess_image(negative_path)
            
        # Yield the batch in the format our model expects
        yield ((anchor_images, positive_images, negative_images),)

# --- SANITY CHECK ---
if __name__ == "__main__":
    
    print("--- Phase 2 (Upgraded): Sanity Check ---")
    
    # 1. Load the data (filepaths)
    (x_paths, y_labels), label_to_indices, class_names = load_data_from_folders(DATASET_DIR)
    
    print(f"\nTotal classes: {NUM_CLASSES}")
    print(f"Total training images: {len(x_paths)}")
    
    # 2. Initialize the generator
    BATCH_SIZE = 16 # Use a smaller batch for testing
    generator = triplet_generator(
        BATCH_SIZE, x_paths, label_to_indices, NUM_CLASSES
    )
    
    # 3. Get one batch of triplets
    print(f"Generating one batch of {BATCH_SIZE} triplets...")
    ((anchor_batch, positive_batch, negative_batch),) = next(generator)
    
    print("Batch generated!")
    print(f"  Anchor batch shape: {anchor_batch.shape}")
    
    # 4. Visualize the first triplet
    import matplotlib.pyplot as plt
    
    # We can't get the label easily without y_labels,
    # so we'll just show the images.
    print(f"\nVisualizing first triplet from batch...")

    plt.figure(figsize=(9, 3))
    plt.suptitle("Triplet Generator Sanity Check (High-Res)")
    
    plt.subplot(1, 3, 1)
    plt.imshow(anchor_batch[0]) # No np.squeeze needed for color
    plt.title(f"Anchor")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(positive_batch[0])
    plt.title(f"Positive")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(negative_batch[0])
    plt.title(f"Negative")
    plt.axis('off')
    
    plt.show()
    
    print("\n--- Phase 2 (Upgraded) Complete ---")