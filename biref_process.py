import torch
import torch.nn.functional as F
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
 
# The portrait model splits images into a 31x31 patch grid internally,
# so both spatial dimensions must be divisible by 31.
PATCH_MULTIPLE = 31
 
def pad_to_multiple(tensor, multiple=PATCH_MULTIPLE):
    """Pad H and W up to the nearest multiple of `multiple` (right/bottom pad)."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        # F.pad order: (left, right, top, bottom)
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, h, w
 
def process_images(input_path, output_path, total):
    open_dir = path(input_path).glob('*.png')
    pbar = tqdm(total=total)
    for img_path in open_dir:
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
 
        input_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        input_tensor = input_tensor.to(device)
 
        # Pad to a size the model can handle, remember original dims to crop back
        padded_tensor, orig_h, orig_w = pad_to_multiple(input_tensor)
 
        with torch.no_grad():
            output = model(padded_tensor)[0]
 
        # Crop the mask back to the original image dimensions
        mask = output.squeeze().cpu().numpy()
        mask = mask[:orig_h, :orig_w]
        mask = (mask > 0.5).astype(np.uint8) * 255
 
        rgba = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = mask
 
        save_path = path(output_path) / img_path.name
        cv2.imwrite(str(save_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
 
        pbar.update(1)
    pbar.close()
