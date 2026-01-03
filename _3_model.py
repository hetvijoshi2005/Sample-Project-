import tensorflow as tf
from tensorflow.keras import layers, Model

# --- Constants ---
# NEW, UPGRADED IMAGE SHAPE!
IMG_SHAPE = (128, 128, 3) 

EMBEDDING_DIM = 128 
TRIPLET_MARGIN = 0.5

# --- Step 3.1: The Base "Embedding" Model (UPGRADED) ---

def get_embedding_model(input_shape, embedding_dim):
    """
    Builds the base model using MobileNetV2 for transfer learning.
    """
    print(f"Building EMBEDDING model with input shape {input_shape}...")
    
    # 1. Load the pre-trained MobileNetV2
    # include_top=False: Don't include the final 1000-class classifier
    # weights='imagenet': Use the weights it learned from ImageNet
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # 2. Freeze the base model
    # We don't want to re-train the entire network, just our new layers.
    base_model.trainable = False
    
    # 3. Build our new "top" layers
    inputs = base_model.input
    
    # Add a pooling layer to flatten the features
    x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    
    # Add a dense layer (our new embedding)
    x = layers.Dense(embedding_dim, name="embedding_vector")(x)
    
    # Add the L2 normalization layer
    embeddings = layers.Lambda(
        tf.math.l2_normalize,
        arguments={'axis': 1},
        output_shape=(embedding_dim,),
        name="l2_normalization"
    )(x)
    
    # 4. Create the final Keras model
    embedding_model = Model(inputs=inputs, outputs=embeddings, name="Embedding_Model")
    
    return embedding_model

# --- Step 3.2 & 3.3: The Full Triplet Network (No Changes Needed) ---
# This logic is the same as our final Fashion-MNIST version

class TripletLossModel(Model):
    """
    A custom Keras Model that wraps the embedding model and implements 
    the triplet loss.
    """
    def __init__(self, embedding_model, margin=0.5, **kwargs):
        super().__init__(**kwargs)
        self.embedding_model = embedding_model
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="triplet_loss")

    def call(self, inputs):
        # 'inputs' from the generator is ((A, P, N),)
        # We unpack the inner tuple 'inputs[0]'
        anchor_input, positive_input, negative_input = inputs[0]
        
        anchor_embeddings = self.embedding_model(anchor_input)
        positive_embeddings = self.embedding_model(positive_input)
        negative_embeddings = self.embedding_model(negative_input)
        return anchor_embeddings, positive_embeddings, negative_embeddings

    def _calculate_loss(self, anchor_emb, positive_emb, negative_emb):
        """
        Calculates the triplet loss.
        """
        positive_distance = tf.reduce_sum(
            tf.square(anchor_emb - positive_emb), axis=-1
        )
        negative_distance = tf.reduce_sum(
            tf.square(anchor_emb - negative_emb), axis=-1
        )
        loss = positive_distance - negative_distance
        loss = tf.maximum(loss + self.margin, 0.0) 
        return tf.reduce_mean(loss) 

    def train_step(self, data):
        # 'data' is ((A, P, N),)
        with tf.GradientTape() as tape:
            # 'self(data, ...)' passes data to 'call()'
            anchor_emb, positive_emb, negative_emb = self(data, training=True)
            loss = self._calculate_loss(anchor_emb, positive_emb, negative_emb)
            
        # We only train the *new* layers (Dense and L2Norm)
        # because we set base_model.trainable = False
        gradients = tape.gradient(loss, self.embedding_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.embedding_model.trainable_variables)
        )
        self.loss_tracker.update_state(loss)
        return {"triplet_loss": self.loss_tracker.result()}

    def test_step(self, data):
        # 'data' is ((A, P, N),)
        anchor_emb, positive_emb, negative_emb = self(data, training=False)
        loss = self._calculate_loss(anchor_emb, positive_emb, negative_emb)
        self.loss_tracker.update_state(loss)
        return {"triplet_loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]

# --- SANITY CHECK ---
if __name__ == "__main__":
    print("--- Phase 3 (Upgraded): Model Architecture ---")
    
    # 1. Build the base embedding model
    base_model = get_embedding_model(IMG_SHAPE, EMBEDDING_DIM)
    
    print("\n--- Base Embedding Model Summary ---")
    base_model.summary()
    
    # 2. Build the full training model
    triplet_model = TripletLossModel(base_model, margin=TRIPLET_MARGIN)
    
    triplet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    )

    print("\n--- Full Triplet Training Model Built ---")
    print(f"  Embedding Dimensions: {EMBEDDING_DIM}")
    print(f"  Triplet Loss Margin:  {TRIPLET_MARGIN}")
    print("We are ready for Phase 4: Model Training.")