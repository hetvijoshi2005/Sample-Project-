import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# --- Configuration ---
INDEX_PATH = "search_index_v2.pkl"
SAMPLE_SIZE = 3000 # We'll plot 3000 points to keep it readable

print("--- Phase 7: t-SNE Visualization ---")

# 1. Load the Index
print(f"Loading data from {INDEX_PATH}...")
df = pd.read_pickle(INDEX_PATH)

# 2. Sample the Data
# Plotting all 33k points is too messy. We'll take a random sample.
if len(df) > SAMPLE_SIZE:
    df_sample = df.sample(SAMPLE_SIZE, random_state=42)
else:
    df_sample = df

print(f"Processing {len(df_sample)} images...")

# Extract embeddings (convert from list to numpy array)
embeddings = np.array(df_sample["embedding"].tolist())

# 3. Simplify Labels
# We have 297 detailed classes (e.g., "Audi TT", "Pizza"). 
# Let's group them into broader categories for the plot colors.
# Note: This logic assumes your food classes are lowercase/snake_case 
# and car classes are Title Case (which is true for these datasets).
def get_broad_category(label):
    if label[0].isupper(): 
        return "Car"
    else:
        return "Food"

df_sample["Category"] = df_sample["label"].apply(get_broad_category)

# 4. Run t-SNE (The Math Magic)
print("Running t-SNE (this might take a minute)...")
tsne = TSNE(n_components=2, verbose=1, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(embeddings)

# Add X and Y coordinates to our dataframe
df_sample["x"] = tsne_results[:, 0]
df_sample["y"] = tsne_results[:, 1]

# 5. Plot!
print("Generating plot...")
plt.figure(figsize=(16, 10))
sns.scatterplot(
    x="x", y="y",
    hue="Category",
    palette="viridis",
    data=df_sample,
    legend="full",
    alpha=0.6
)

plt.title("t-SNE Visualization of Image Embeddings", fontsize=20)
plt.xlabel("Dimension 1", fontsize=12)
plt.ylabel("Dimension 2", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.3)

# Save the plot
save_path = "tsne_visualization.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"✅ Visualization saved to {save_path}")

# Show plot
plt.show()