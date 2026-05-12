from rembg import remove, new_session
from PIL import Image, ImageFilter
import numpy as np

session = new_session("u2net_human_seg")
image_path = "../full_PNG_export/DSC03427.png"
img = Image.open(image_path).convert("RGB")
result = remove(
    img,
    session=session,
    alpha_matting=True,                        # enables proper edge matting
    alpha_matting_foreground_threshold=240,    # pixels above this = definitely foreground
    alpha_matting_background_threshold=10,     # pixels below this = definitely background
    alpha_matting_erode_size=10,               # how much to shrink the mask before matting
)

result.save("output/portrait_no_bg.png")