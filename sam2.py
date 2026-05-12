from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.misc import variant_to_config_mapping

image_path = "portrait.jpg"
img = Image.open(image_path).convert("RGB")
img_array = np.array(img)

h, w = img_array.shape[:2]

# --- Build model ---
model = build_sam2(
    variant_to_config_mapping["large"],
    "sam2_hiera_large.pt",
)
image_predictor = SAM2ImagePredictor(model)
image_predictor.set_image(img_array)

# --- Point prompts centered on the person ---
# Positive points: spread across the center of the image (face, torso)
input_point = np.array([
    [w // 2, h // 4],       # head area
    [w // 2, h // 2],       # torso
    [w // 2, int(h * 0.7)], # lower torso
])
input_label = np.array([1, 1, 1])  # 1 = foreground

# Negative points: corners of the image (background)
neg_points = np.array([
    [10,     10],
    [w - 10, 10],
    [10,     h - 10],
    [w - 10, h - 10],
])
neg_labels = np.array([0, 0, 0, 0])  # 0 = background

all_points = np.concatenate([input_point, neg_points], axis=0)
all_labels = np.concatenate([input_label, neg_labels], axis=0)

# --- Predict ---
masks, scores, logits = image_predictor.predict(
    point_coords=all_points,
    point_labels=all_labels,
    box=None,
    multimask_output=True,
)

# --- Pick the best mask ---
best_mask = masks[np.argmax(scores)]  # shape: (H, W), dtype: bool

# --- Apply mask: make background transparent ---
rgba = np.dstack([img_array, np.zeros((h, w), dtype=np.uint8)])  # add alpha channel
rgba[..., 3] = (best_mask * 255).astype(np.uint8)                # set alpha from mask

result = Image.fromarray(rgba, mode="RGBA")
result.save("portrait_no_bg.png")  # PNG preserves transparency

# --- Preview ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img_array)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(result)
axes[1].set_title("Background Removed")
axes[1].axis("off")

plt.tight_layout()
plt.show()