import tensorflow as tf
import numpy as np
import os

# --- GPU Setup and Verification (Add this at the very top) ---
print("="*60)
print("GPU CONFIGURATION")
print("="*60)

# Check TensorFlow and GPU availability
print(f"TensorFlow Version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# Configure GPU memory growth (prevents OOM errors)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for: {gpu}")
        
        # Print detailed GPU info
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        print(f"✓ GPU Device: {gpu_details.get('device_name', 'Unknown')}")
        
    except RuntimeError as e:
        print(f"✗ GPU configuration error: {e}")
else:
    print("✗ WARNING: No GPU found! Training will be very slow on CPU.")
    print("Please check your CUDA installation.")

# Optional: Enable mixed precision for faster training
# Uncomment the next 2 lines if you want even faster training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("✓ Mixed precision (FP16) enabled")

print("="*60)
print()

# --- Import our code from other files ---
# Make sure your files are named with the underscore
from _2_triplet_generator import load_data_from_folders, triplet_generator, DATASET_DIR, IMG_SHAPE
from _3_model import get_embedding_model, TripletLossModel, EMBEDDING_DIM, TRIPLET_MARGIN

# --- Configuration ---
# !! This MUST be run on a GPU !!
# If Colab crashes with "Out of Memory" (OOM), lower this to 16
BATCH_SIZE = 128 
# 5 epochs is a good start for fine-tuning a pre-trained model
EPOCHS = 5      
# We'll run 200 batches per epoch.
# This is enough to learn without taking all day.
STEPS_PER_EPOCH = 500 
LEARNING_RATE = 0.0001
# We'll save this as a new model file
EMBEDDING_MODEL_SAVE_PATH = "embedding_model_v2.keras" 

# --- Main Training Script ---
if __name__ == "__main__":
    
    print("--- Phase 4 (Upgraded): Model Training ---")
    
    # 1. Load Data (Filepaths)
    (x_train_paths, _), label_to_indices, class_names = load_data_from_folders(DATASET_DIR)
    NUM_CLASSES = len(class_names)
    print(f"Data filepaths loaded. Found {NUM_CLASSES} classes.")
    
    # 2. Create the Triplet Generator
    train_generator = triplet_generator(
        BATCH_SIZE, x_train_paths, label_to_indices, NUM_CLASSES
    )
    print("Triplet generator created.")
    
    # 3. Build the Model
    # This will load the MobileNetV2 model
    base_embedding_model = get_embedding_model(IMG_SHAPE, EMBEDDING_DIM)
    
    triplet_model = TripletLossModel(
        base_embedding_model, margin=TRIPLET_MARGIN
    )
    
    triplet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    )
    print("Model built and compiled.")
    
    # 4. Train the Model
    print(f"Starting training for {EPOCHS} epochs (on GPU)...")
    print("TIP: Open a new terminal and run 'watch -n 1 nvidia-smi' to monitor GPU usage")
    print()
    
    try:
        history = triplet_model.fit(
            train_generator,
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=EPOCHS,
            verbose=1
        )
        print("\nTraining complete!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
    
    # 5. Save the Base Embedding Model
    print(f"\nSaving base embedding model to {EMBEDDING_MODEL_SAVE_PATH}...")
    base_embedding_model.save(EMBEDDING_MODEL_SAVE_PATH)
    
    print("\n--- Phase 4 (Upgraded) Complete ---")
    print(f"Model saved to {EMBEDDING_MODEL_SAVE_PATH}")
    print("\nNext Steps:")
    print("1. Check the training loss - it should decrease over epochs")
    print("2. Proceed to Phase 5 to build the search index")
    print("3. Use the saved model for inference")