from PIL import Image
import os

# --- Config ---
input_path = "gan_results/clean_epoch10.png"  # Change as needed
output_dir = "gan_split/clean256"
prefix = "clean10"  # Short for "generated_epoch10"
grid_size = 4  # 4x4 grid
tile_size = 256  # each tile is 256x256

os.makedirs(output_dir, exist_ok=True)

# --- Load and Split ---
img = Image.open(input_path).convert("RGB")

for row in range(grid_size):
    for col in range(grid_size):
        left = col * tile_size
        upper = row * tile_size
        right = left + tile_size
        lower = upper + tile_size

        tile = img.crop((left, upper, right, lower))
        idx = row * grid_size + col
        tile.save(os.path.join(output_dir, f"{prefix}_{idx:02}.png"))

print(f"✅ Split complete. Saved to: {output_dir}")