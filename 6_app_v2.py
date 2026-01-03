import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI Visual Search", layout="wide")

# --- 2. DEFINE CUSTOM FUNCTIONS ---
# This fixes the "Unknown layer: Lambda" error
def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=1)

# --- 3. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        # A. Load Model (With Custom Object)
        model = tf.keras.models.load_model(
            'embedding_model_v2.keras', 
            custom_objects={'l2_normalize': l2_normalize}
        )
        
        # B. Load Index (Dual-Mode Logic)
        full_index_path = 'search_index_v2.pkl'
        mini_index_path = 'mini_index.pkl'
        df = None
        
        # Check for Local Pro Mode (Full Index + Dataset folder)
        if os.path.exists(full_index_path) and os.path.exists("dataset"):
            print("✅ Loading FULL Local Index (Pro Mode)...")
            df = pd.read_pickle(full_index_path)
            mode = "PRO"
            
        # Check for Cloud Demo Mode
        elif os.path.exists(mini_index_path):
            print("☁️ Loading LITE Cloud Index (Demo Mode)...")
            df = pd.read_pickle(mini_index_path)
            mode = "LITE"
            
        else:
            st.error("🚨 CRITICAL ERROR: No index file found! (Checked for 'search_index_v2.pkl' and 'mini_index.pkl')")
            return None, None, None

        return model, df, mode
        
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None

# --- 4. MAIN APPLICATION ---
def main():
    st.title("🍔🏎️ AI Similarity Search: Cars & Food")
    st.write("Upload an image to find similar items from our database.")

    # Load everything
    model, df, mode = load_resources()

    if model is None or df is None:
        st.stop()

    # Show Mode Badge
    if mode == "LITE":
        st.warning("⚠️ **DEMO MODE ACTIVE:** Searching a curated subset of 25 popular classes. (Clone repo for full 297-class Pro Mode).")
    else:
        st.success("✅ **PRO MODE ACTIVE:** Searching full database (297 Classes).")

    # File Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display Query Image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption='Query Image', use_container_width=True)
        
        with col2:
            st.write("🔍 **Analyzing...**")
            
            try:
                # --- SMART PREPROCESSING (Fixes Shape & Type Errors) ---
                
                # 1. Get Expected Input Shape from Model
                # Usually (None, 224, 224, 3)
                input_shape = model.input_shape
                
                # Handle generic lists
                if isinstance(input_shape, list):
                    input_shape = input_shape[0]
                
                # Default to 224 if model doesn't specify
                target_h = input_shape[1] if input_shape[1] is not None else 224
                target_w = input_shape[2] if input_shape[2] is not None else 224
                
                # 2. Resize Image
                img_resized = image.resize((target_w, target_h))
                
                # 3. Convert to Array and Normalize
                img_array = np.array(img_resized) / 255.0
                
                # 4. FORCE FLOAT32 (Crucial for Keras 3 / TensorFlow)
                img_array = img_array.astype(np.float32)
                
                # 5. Add Batch Dimension (1, H, W, 3)
                img_array = np.expand_dims(img_array, axis=0)
                
                # --- PREDICTION & SEARCH ---
                
                # Get Embedding
                query_embedding = model.predict(img_array)
                
                # Search (Cosine Similarity)
                database_embeddings = np.stack(df['embedding'].values)
                similarities = cosine_similarity(query_embedding, database_embeddings)
                
                # Get Top 5 Results
                top_k = 5
                top_indices = np.argsort(similarities[0])[::-1][:top_k]
                
                st.write(f"✅ Found {top_k} matches:")

                # Display Results
                st.divider()
                cols = st.columns(5)
                
                for i, idx in enumerate(top_indices):
                    row = df.iloc[idx]
                    match_path = row['filepath']
                    label = row['label']
                    score = similarities[0][idx]
                    
                    with cols[i]:
                        # IMAGE LOADING LOGIC
                        display_path = match_path
                        
                        # Fix path for Cloud Demo Mode
                        if mode == "LITE":
                            filename = os.path.basename(match_path)
                            display_path = os.path.join("app_images", filename)
                        
                        # Display
                        if os.path.exists(display_path):
                            st.image(display_path, caption=f"{label}\n({score:.2f})")
                        else:
                            st.error(f"Image missing: {label}")

            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.write(f"Debug: Model expects shape {model.input_shape}")

if __name__ == "__main__":
    main()