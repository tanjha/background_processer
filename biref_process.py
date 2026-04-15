import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
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

_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
_to_tensor = transforms.ToTensor()

_model = None


def _inference_size(orig_h, orig_w, ref=1024):
    """Scale longest edge to ref, keeping both dims divisible by 32 (MODNet requirement)."""
    scale = min(ref / max(orig_h, orig_w), 1.0)
    new_h = max(32, int(orig_h * scale) // 32 * 32)
    new_w = max(32, int(orig_w * scale) // 32 * 32)
    return new_h, new_w


def _get_model():
    global _model
    if _model is None:
        print("Loading MODNet (photographic portrait matting)...")
        _model = torch.hub.load(
            "ZHKKKe/MODNet",
            "modnet_photographic_portrait_matting",
            pretrained=True,
            trust_repo=True,
        )
        _model.to(device)
        _model.eval()
        torch.cuda.synchronize()
        print("Model ready.\n")
    return _model


def process_images(input_path, output_path):
    model = _get_model()
    image_paths = list(path(input_path).glob("*.png"))
    path(output_path).mkdir(parents=True, exist_ok=True)

    pbar = tqdm(image_paths, file=sys.stdout)
    for img_path in pbar:
        pbar.set_description(img_path.name[:40])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size  # PIL size is (width, height)

        # Resize to a MODNet-compatible resolution (both dims divisible by 32)
        inf_h, inf_w = _inference_size(orig_h, orig_w)
        resized = image.resize((inf_w, inf_h), Image.LANCZOS)

        img_tensor = _normalize(_to_tensor(resized)).unsqueeze(0).to(device)

        with torch.no_grad():
            # MODNet returns (semantic, detail, matte); inference=True skips training branches
            _, _, matte = model(img_tensor, inference=True)

        # matte: [1, 1, H, W], values already in [0, 1] — no sigmoid needed
        matte_np = matte.squeeze().cpu().numpy()
        alpha = cv2.resize(matte_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        alpha = (alpha * 255).astype(np.uint8)

        image_np = np.array(image)
        rgba = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = alpha

        save_path = path(output_path) / img_path.name
        cv2.imwrite(str(save_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


if __name__ == "__main__":
    process_images("input", "output")
