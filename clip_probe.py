import glob, os, random
import numpy as np
import torch
import joblib
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.linear_model import LogisticRegression

random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NAME = "openai/clip-vit-large-patch14"
IMG = (".jpg", ".jpeg", ".png", ".webp")
BASE = "faces/real_vs_fake/real-vs-fake"
N_BASE = 1500

dtype = torch.float16 if DEV == "cuda" else torch.float32
model = CLIPModel.from_pretrained(NAME, torch_dtype=dtype).to(DEV).eval()
proc = CLIPProcessor.from_pretrained(NAME)


def files(d, n=None):
    fs = [f for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)
          if f.lower().endswith(IMG)]
    if n and len(fs) > n:
        fs = random.sample(fs, n)
    return fs


@torch.no_grad()
def feats(paths, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        ims = []
        for p in paths[i:i + bs]:
            try:
                ims.append(Image.open(p).convert("RGB"))
            except Exception:
                ims.append(Image.new("RGB", (224, 224)))
        x = proc(images=ims, return_tensors="pt")["pixel_values"].to(DEV, dtype)
        f = model.get_image_features(pixel_values=x)
        if not torch.is_tensor(f):
            f = f.pooler_output
        f = f.float()
        f = f / f.norm(dim=-1, keepdim=True)
        out.append(f.cpu().numpy())
        if (i // bs) % 20 == 0:
            print(f"  features {min(i + bs, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(out)


# ---- training data: benchmark extras + a sample of the 140k set ----
fake = files("extra/fake") + files(os.path.join(BASE, "train", "fake"), N_BASE)
real = files("extra/real") + files(os.path.join(BASE, "train", "real"), N_BASE)
print(f"training images: fake={len(fake)} real={len(real)}")
X = np.concatenate([feats(fake), feats(real)])
y = np.array([1] * len(fake) + [0] * len(real))   # 1 = fake

clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
clf.fit(X, y)
os.makedirs("models", exist_ok=True)
joblib.dump({"clf": clf, "clip": NAME}, "models/clip_probe.joblib")
print("saved models/clip_probe.joblib")

# ---- evaluation on held-out folders ----
print("\n=== dataset test sample (100 each) ===")
for cls in ("fake", "real"):
    fs = files(os.path.join(BASE, "test", cls), 100)
    p = clf.predict_proba(feats(fs))[:, 1]
    print(f"test/{cls}: mean p_fake={p.mean():.3f}  flagged fake={(p > 0.5).mean() * 100:.0f}%")

print("\n=== your held-out folders ===")
for d in sorted(glob.glob("images/*/")):
    fs = files(d)
    if not fs:
        continue
    p = clf.predict_proba(feats(fs))[:, 1]
    print(f"{os.path.normpath(d):22s} n={len(fs):3d}  mean p_fake={p.mean():.3f}  "
          f"flagged fake={(p > 0.5).mean() * 100:.0f}%")
    if len(fs) <= 5:
        for f, v in zip(fs, p):
            print(f"      {os.path.basename(f):28s} p_fake={v:.3f}")