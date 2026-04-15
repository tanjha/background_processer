import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoModelForImageSegmentation
from pathlib import Path as path
from tqdm import tqdm
 
device = "cuda" if torch.cuda.is_available() else "cpu"
 
model = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet-portrait",
    trust_remote_code=True
)
 
model.to(device)
model.eval()
 
# BiRefNet-portrait was trained at this resolution.
# Input must be resized to this size before inference.
MODEL_SIZE = (1024, 1024)  # (width, height)
 
def process_images(input_path, output_path, total):
    open_dir = path(input_path).glob('*.png')
    pbar = tqdm(total=total)
    for img_path in open_dir:
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size  # PIL size is (width, height)
 
        # Resize to the model's expected input size
        resized = image.resize(MODEL_SIZE, Image.LANCZOS)
        image_np = np.array(resized)
 
        input_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        input_tensor = input_tensor.to(device)
 
        with torch.no_grad():
            output = model(input_tensor)[0]
 
        # output mask is at model resolution — resize back to original
        mask = output.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 127).astype(np.uint8) * 255  # re-threshold after resize
 
        # Apply mask to original full-resolution image
        orig_np = np.array(image)
        rgba = cv2.cvtColor(orig_np, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = mask
 
        save_path = path(output_path) / img_path.name
        cv2.imwrite(str(save_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
 
        pbar.update(1)
    pbar.close()