import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from pathlib import Path as path
from tqdm import tqdm

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA not available — PyTorch cannot see your GPU.\n"
        f"  torch version : {torch.__version__}\n"
        f"  torch CUDA build: {torch.version.cuda}\n"
        "Fix: reinstall PyTorch with CUDA support, e.g.:\n"
        "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
        "(use cu128 for RTX 5090 / Blackwell; cu126 for older cards)"
    )

device = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}")

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model = None

def _get_model():
    """Load the model on first call, then return the cached instance."""
    global _model
    if _model is None:
        print(f"Loading model onto {device}...")
        _model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet_dynamic",
            trust_remote_code=True,
        )
        _model.to(device)
        _model.eval()
        torch.cuda.synchronize()  # ensure model is fully resident on GPU before first inference
        print("Model ready.\n")
    return _model


def process_images(input_path, output_path):
    model = _get_model()
    image_paths = list(path(input_path).glob('*.png'))
    path(output_path).mkdir(parents=True, exist_ok=True)

    model_dtype = next(model.parameters()).dtype

    pbar = tqdm(image_paths, file=sys.stdout)
    for img_path in pbar:
        pbar.set_description(img_path.name[:40])
        image = Image.open(img_path).convert("RGB")
        orig_size = image.size  # (width, height)
        image_np = np.array(image)

        input_tensor = transform(image).unsqueeze(0).to(device=device, dtype=model_dtype)

        with torch.no_grad():
            preds = model(input_tensor)

        # BiRefNet returns a list of predictions (coarse→fine); take the last (finest) scale.
        # sigmoid converts raw logits to [0,1] probabilities — required for BiRefNet_dynamic.
        mask = preds[-1][0].sigmoid().squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255

        # Resize mask back to original image dimensions
        mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_LINEAR)

        rgba = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = mask

        save_path = path(output_path) / img_path.name
        cv2.imwrite(str(save_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


if __name__ == "__main__":
    process_images("input", "output")
