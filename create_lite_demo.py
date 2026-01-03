import os
import shutil
import pandas as pd
import numpy as np

# --- Configuration ---
SOURCE_INDEX = "search_index_v2.pkl"
DEST_DIR = "app_images"
DEST_INDEX = "mini_index.pkl"

# 25 Diverse & Representative Classes (Approved Strategy)
# We use partial names (e.g., "Ferrari") to ensure we match the specific year/model in your dataset.
TARGET_CLASSES = [
    # --- FOOD (12 items: Diverse textures, colors, and shapes) ---
    "pizza",            # Round, colorful, iconic
    "hamburger",        # Stacked layers, distinct shape
    "ice_cream",        # Scoops, soft texture
    "sushi",            # Small, detailed, distinct pattern
    "french_fries",     # Yellow, stick-like
    "fried_rice",       # Grainy texture, bowl
    "hot_dog",          # Cylindrical, distinct bun
    "donuts",           # Ring shape, colorful
    "tacos",            # Shell shape
    "chocolate_cake",   # Dark color, slice shape
    "steak",            # Meat texture, organic shape
    "bibimbap",         # Complex, colorful bowl (shows detail handling)

    # --- CARS (13 items: Diverse body types and angles) ---
    "Ferrari 458",      # Supercar, sleek, often Red/Yellow
    "Lamborghini",      # Sharp angles, distinct silhouette
    "Hummer",           # Boxy, Large SUV (Contrast to sleek cars)
    "Jeep Wrangler",    # Off-road, rugged shape
    "Fiat 500",         # Very small, compact (Contrast to Hummer)
    "Audi TT",          # Rounded coupe, distinct curves
    "BMW M3",           # Classic sedan/coupe profile
    "Chevrolet Corvette", # American sports car, long hood
    "Dodge Challenger", # Muscle car, blocky front
    "Ford Mustang",     # Muscle car, iconic grille
    "Nissan",      # Japanese sports car, distinct tail lights
    "Toyota Camry",     # Standard sedan (Baseline for "normal" cars)
    "Volkswagen Golf"   # Hatchback (Distinct rear shape)
]

IMAGES_PER_CLASS = 20  # 25 classes * 20 images = 500 images total (Approx 25MB)

print(f"--- Creating Curated Demo Dataset ({len(TARGET_CLASSES)} Classes) ---")

# 1. Load the full index
if not os.path.exists(SOURCE_INDEX):
    print(f"❌ Error: {SOURCE_INDEX} not found. Are you in the right folder?")
    exit()

df = pd.read_pickle(SOURCE_INDEX)
print(f"Loaded full index with {len(df)} images.")

# 2. Filter logic
df_mini = pd.DataFrame()

print("\nScanning dataset for diverse classes...")
found_classes = 0

for target in TARGET_CLASSES:
    # Smart Match: Look for the target string inside the label (Case Insensitive)
    # This handles "Ferrari 458 Italia 2012" when we search "Ferrari 458"
    matches = df[df['label'].str.contains(target, case=False, regex=False)]
    
    if len(matches) > 0:
        # If multiple classes match (e.g. "BMW M3 Coupe" and "BMW M3 Sedan"), 
        # we pick the one with the most images to be safe.
        best_class = matches['label'].value_counts().idxmax()
        specific_matches = matches[matches['label'] == best_class]
        
        # Sample images
        count = min(len(specific_matches), IMAGES_PER_CLASS)
        sample = specific_matches.sample(count, random_state=42)
        
        df_mini = pd.concat([df_mini, sample])
        print(f"  ✅ Found: '{best_class}' (Added {count} images)")
        found_classes += 1
    else:
        print(f"  ⚠️ Warning: Could not find any class matching '{target}'")

# 3. Create destination folder
if os.path.exists(DEST_DIR):
    shutil.rmtree(DEST_DIR)
os.makedirs(DEST_DIR)

print(f"\nCopying {len(df_mini)} images to '{DEST_DIR}'...")

new_filepaths = []
successful_copies = 0

# 4. Copy images physically
for index, row in df_mini.iterrows():
    old_path = row['filepath']
    
    # Check if the file actually exists on your laptop before copying
    if os.path.exists(old_path):
        filename = os.path.basename(old_path)
        new_path = os.path.join(DEST_DIR, filename)
        shutil.copy(old_path, new_path)
        new_filepaths.append(new_path)
        successful_copies += 1
    else:
        # If the image is missing from your drive, skip it
        new_filepaths.append(None)

# 5. Save the Mini Index
df_mini['filepath'] = new_filepaths
df_mini = df_mini.dropna(subset=['filepath']) # Remove broken links

# Save ONLY the columns we need
df_mini[['label', 'filepath', 'embedding']].to_pickle(DEST_INDEX)

print("-" * 40)
print(f"🎉 SUCCESS! Lite Dataset Ready.")
print(f"   - Classes: {found_classes}/{len(TARGET_CLASSES)}")
print(f"   - Images:  {successful_copies}")
print(f"   - Folder:  {DEST_DIR}/")
print(f"   - Index:   {DEST_INDEX}")
print("\n👉 NEXT STEP: Upload 'app_images' and 'mini_index.pkl' to GitHub.")