from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

root = tk.Tk()
root.withdraw()
input_image_path = filedialog.askopenfilename(
    title="Select a photo",
    filetypes=[("Image files", "*.png *.jpg *.jpeg *.tiff *.bmp *.webp")]
)
if not input_image_path:
    raise SystemExit("No file selected.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForImageSegmentation.from_pretrained(
    'briaai/RMBG-2.0', trust_remote_code=True
).eval().to(device)

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image = Image.open(input_image_path).convert("RGB")
orig_w, orig_h = image.size

# Pad to square so the model sees an undistorted image.
# Squashing a 3:2 portrait to 1:1 stretches the subject and degrades edge quality.
sq_size = max(orig_w, orig_h)
pad_l = (sq_size - orig_w) // 2
pad_t = (sq_size - orig_h) // 2
padded = Image.new("RGB", (sq_size, sq_size), (127, 127, 127))
padded.paste(image, (pad_l, pad_t))

input_tensor = transform_image(padded).unsqueeze(0).to(device)

with torch.no_grad():
    preds = model(input_tensor)[-1].sigmoid().cpu()

# Upscale the mask to the padded square size, then crop the padding back out.
# LANCZOS keeps edges sharp at native resolution.
mask_pil = transforms.ToPILImage()(preds[0].squeeze())
mask_sq = mask_pil.resize((sq_size, sq_size), Image.LANCZOS)
mask = mask_sq.crop((pad_l, pad_t, pad_l + orig_w, pad_t + orig_h))

image.putalpha(mask)

out_path = Path(input_image_path).with_stem(Path(input_image_path).stem + "_nobg").with_suffix(".png")
image.save(out_path)
print(f"Saved: {out_path}")
