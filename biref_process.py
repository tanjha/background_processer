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

def process_images(input_path, output_path, total):
    open_dir = path(input_path).glob('*.png')
    pbar = tqdm(total=total)
    for img_path in open_dir:
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        input_tensor = torch.from_numpy(image_np).permute(2,0,1).unsqueeze(0).float() / 255.0
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)[0]

        mask = output.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255

        rgba = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
        rgba[:,:,3] = mask

        save_path = path(output_path) / img_path.name
        cv2.imwrite(save_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

        pbar.update(1)
    pbar.close()