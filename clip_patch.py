import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import math
import numpy as np
import torch
import joblib
from PIL import Image
import pipeline as p

THRESH = float(os.environ.get("CLIP_THRESH", 0.60))
PROBE = os.path.join(p.MODEL_DIR, "clip_probe.joblib")
_state = {}


def _load():
    if "clf" in _state:
        return _state
    from transformers import CLIPModel, CLIPProcessor
    d = joblib.load(PROBE)
    dtype = torch.float16 if p.device.type == "cuda" else torch.float32
    _state["clf"] = d["clf"]
    _state["model"] = CLIPModel.from_pretrained(d["clip"], dtype=dtype).to(p.device).eval()
    _state["proc"] = CLIPProcessor.from_pretrained(d["clip"])
    _state["dtype"] = dtype
    print(f"[CLIP] probe loaded, threshold={THRESH}")
    return _state


@torch.no_grad()
def _raw_probs(pil_images, bs=16):
    s = _load()
    out = []
    for i in range(0, len(pil_images), bs):
        x = s["proc"](images=pil_images[i:i + bs], return_tensors="pt")["pixel_values"]
        f = s["model"].get_image_features(pixel_values=x.to(p.device, s["dtype"]))
        if not torch.is_tensor(f):
            f = f.pooler_output
        f = f.float()
        f = f / f.norm(dim=-1, keepdim=True)
        out.append(s["clf"].predict_proba(f.cpu().numpy())[:, 1])
    return np.concatenate(out)


def _logit(x):
    x = min(max(float(x), 1e-6), 1 - 1e-6)
    return math.log(x / (1 - x))


def _adjust(pf):
    """Shift so that raw p_fake == THRESH maps to 0.5."""
    return 1.0 / (1.0 + math.exp(-(_logit(pf) - _logit(THRESH))))


def _score_image(image):
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        raw = _raw_probs([image.convert("RGB")])[0]
        fake = _adjust(raw)
        return 1.0 - fake, fake, True
    except Exception as e:
        print(f"[CLIP] image error: {e}")
        return None


def _score_video(video_path, n_samples=30):
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None
        wanted = set(np.linspace(0, total - 1, n_samples, dtype=int).tolist())
        face_imgs, all_imgs = [], []
        idx = 0
        while True:
            if not cap.grab():
                break
            if idx in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    crop = p._crop_face(rgb)
                    if crop is not None and crop.size > 0:
                        im = Image.fromarray(crop)
                        face_imgs.append(im)
                    else:
                        im = Image.fromarray(rgb)
                    all_imgs.append(im)
            idx += 1
            if idx > max(wanted):
                break
        cap.release()
        use = face_imgs if face_imgs else all_imgs
        if not use:
            return None
        fake = float(np.mean([_adjust(v) for v in _raw_probs(use)]))
        return 1.0 - fake, fake, len(all_imgs), len(face_imgs)
    except Exception as e:
        print(f"[CLIP] video error: {e}")
        return None


if os.path.exists(PROBE):
    p._score_image = _score_image
    p._score_video = _score_video
    print("[CLIP] image + video scoring patched to use the CLIP probe")
else:
    print(f"[CLIP] {PROBE} not found, using the original models")