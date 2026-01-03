import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. DEFINE CLASS NAMES
# We'll use this later for our filtering feature
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
num_classes = len(class_names)

# 2. LOAD THE FASHION-MNIST DATASET
print("Loading Fashion-MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 3. PRE-PROCESS THE IMAGES
def preprocess_images(images):
    # Normalize pixel values from 0-255 to 0.0-1.0
    images = images.astype('float32') / 255.0
    # Add a channel dimension (28, 28) -> (28, 28, 1)
    images = np.expand_dims(images, axis=-1)
    return images

x_train = preprocess_images(x_train)
x_test = preprocess_images(x_test)

print(f"Training data shape: {x_train.shape} (Images, Height, Width, Channels)")
print(f"Training labels shape: {y_train.shape} (Labels,)")

# 4. CREATE LABEL-TO-INDICES MAPPING (FOR TRIPLET MINING)
# This is the most important step for Phase 2.
# We need this to quickly find a "Positive" (same class) and
# "Negative" (different class) image.

print("\nCreating label-to-index mappings...")
label_to_indices = {label: np.where(y_train == label)[0]
                    for label in range(num_classes)}

# Print a summary to check our work
print(f"Created {len(label_to_indices)} class mappings.")
for i in range(num_classes):
    print(f"  Class {i} ({class_names[i]}): {len(label_to_indices[i])} images")

# 5. (OPTIONAL) VISUALIZE A FEW IMAGES
print("\nShowing a few sample images...")
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # Note: np.squeeze removes the (1) channel dim for plotting
    plt.imshow(np.squeeze(x_train[i]), cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()

print("\n--- Phase 1 Complete ---")
print("We are ready for Phase 2: Building the Triplet Data Pipeline.")
print("We will use 'x_train', 'y_train', and 'label_to_indices' in the next step.")

# We will need these variables for the next phases.
# For now, just running this script is enough.