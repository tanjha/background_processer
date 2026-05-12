from rembg import remove, new_session
from PIL import Image, ImageFilter
import numpy as np

session = new_session("u2net_human_seg")
image_path = "../full_PNG_export/DSC03427.png"
img = Image.open(image_path).convert("RGB")
result = remove(img, session=session)

# Extract and sharpen the alpha channel
r, g, b, alpha = result.split()

# Push soft edges toward hard 0/255
alpha_np = np.array(alpha, dtype=np.float32)
# Increase contrast of the mask edges
contrast_mult = 5
alpha_np = np.clip((alpha_np - 128) * contrast_mult + 128, 0, 255).astype(np.uint8)

# Optional: slight erode to clean up fringe pixels
alpha_img = Image.fromarray(alpha_np)
alpha_img = alpha_img.filter(ImageFilter.MinFilter(3))  # erodes 1px, removes fringe

result = Image.merge("RGBA", (r, g, b, alpha_img))
result.save("output/portrait_no_bg.png")
print("Done")