import os, glob, random
import numpy as np
import torch
from PIL import Image
import pipeline as p

random.seed(0)
model = p._load_image_model()
ROOT = "faces/real_vs_fake/real-vs-fake/test"
N = 100


def p_fake(img):
    t = p._img_transform(img.convert("RGB")).unsqueeze(0).to(p.device)
    with torch.no_grad():
        return torch.softmax(model(t), 1)[0, 0].item()   # index 0 = fake


def sample(cls):
    files = glob.glob(os.path.join(ROOT, cls, "*"))
    return random.sample(files, min(N, len(files)))


for cls in ("fake", "real"):
    ps = np.array([p_fake(Image.open(f)) for f in sample(cls)])
    flagged = (ps > 0.5).mean() * 100
    print(f"dataset test/{cls}: mean p_fake={ps.mean():.3f}  flagged fake={flagged:.1f}%")

print("\nYour files (p_fake): original size  |  downscaled to 256px")
groups = {}
for f in sorted(glob.glob("images/**/*.*", recursive=True)):
    try:
        img = Image.open(f).convert("RGB")
    except Exception:
        continue
    small = img.resize((256, 256), Image.LANCZOS)
    a, b = p_fake(img), p_fake(small)
    print(f"{f:40s} {a:.3f}  |  {b:.3f}")
    folder = os.path.dirname(f)
    groups.setdefault(folder, []).append(a)

print("\nSummary per folder (original size):")
for folder, ps in groups.items():
    ps = np.array(ps)
    print(f"{folder:25s} n={len(ps):3d}  mean p_fake={ps.mean():.3f}  flagged fake={(ps > 0.5).mean() * 100:.0f}%")