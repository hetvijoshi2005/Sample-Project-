import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm  # Progress bar

# --- Configuration ---
DATASET_DIR = "dataset"
MODEL_PATH = "embedding_model_v2.keras"
INDEX_SAVE_PATH = "search_index_v2.pkl"
IMG_SHAPE = (128, 128, 3)
BATCH_SIZE = 64

def load_test_filepaths(data_dir):
    """
    Scans the dataset/test folder to get all image paths and labels.
    """
    test_dir = os.path.join(data_dir, "test")
    print(f"Scanning {test_dir}...")
    
    file_paths = []
    labels = []
    
    # Get class names (sorted to ensure consistency)
    class_names = sorted(os.listdir(test_dir))
    
    for class_name in tqdm(class_names, desc="Classes"):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_name)
                file_paths.append(img_path)
                labels.append(class_name)
                
    return file_paths, labels, class_names

def preprocess_image(file_path):
    """
    TF-based preprocessing function to be used in tf.data.Dataset
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SHAPE[0], IMG_SHAPE[1]])
    img = img / 255.0  # Normalize to 0-1
    return img

# --- Main Script ---
if __name__ == "__main__":
    print("--- Phase 5 (Upgraded): Building Search Index ---")
    
    # 1. GPU Setup (Optional but good for safety)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU enabled: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(e)
            
    # 2. Load Filepaths
    file_paths, labels, class_names = load_test_filepaths(DATASET_DIR)
    print(f"\nFound {len(file_paths)} test images across {len(class_names)} classes.")
    
    # 3. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    # We need to tell Keras about our custom L2 normalization layer
    custom_objects = {'l2_normalize': tf.math.l2_normalize}
    
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        safe_mode=False, 
        custom_objects=custom_objects
    )
    print("Model loaded successfully.")
    
    # 4. Create Data Pipeline (The "Pro" Way)
    # This loads images on the fly, batches them, and feeds them to the GPU.
    print("Creating data pipeline...")
    
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    image_ds = path_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    image_ds = image_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # 5. Generate Embeddings
    print("Generating embeddings (this may take a few minutes)...")
    embeddings = model.predict(image_ds, verbose=1)
    
    print(f"Embeddings generated. Shape: {embeddings.shape}")
    
    # 6. Save Index
    print(f"Saving index to {INDEX_SAVE_PATH}...")
    
    # We create a simple DataFrame
    # Note: We store the 'filepath' so the App knows which image to display later!
    df = pd.DataFrame({
        "label": labels,
        "filepath": file_paths,
        "embedding": list(embeddings)
    })
    
    df.to_pickle(INDEX_SAVE_PATH)
    
    print("\n--- Phase 5 Complete ---")
    print(f"Saved {len(df)} items to the index.")
    print("Ready for Phase 6 (The App)!")