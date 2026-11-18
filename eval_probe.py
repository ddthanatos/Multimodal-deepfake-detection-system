import glob, os
import numpy as np
import torch
import joblib
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEV = "cuda" if torch.cuda.is_available() else "cpu"
d = joblib.load("models/clip_probe.joblib")
clf, NAME = d["clf"], d["clip"]
dtype = torch.float16 if DEV == "cuda" else torch.float32
model = CLIPModel.from_pretrained(NAME, dtype=dtype).to(DEV).eval()
proc = CLIPProcessor.from_pretrained(NAME)
IMG = (".jpg", ".jpeg", ".png", ".webp")
FAKE_DIRS = ["images/gan", "images/gan_hf", "images/other_fake"]
REAL_DIRS = ["images/real", "images/real_hf"]


def files(folder):
    return [f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
            if f.lower().endswith(IMG)]


@torch.no_grad()
def p_fake(paths, bs=16):
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
        out.append(clf.predict_proba(f.cpu().numpy())[:, 1])
    return np.concatenate(out) if out else np.array([])


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    ranks = np.empty(len(s))
    ranks[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1, n0 = len(pos), len(neg)
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


scores = {}
for d_ in FAKE_DIRS + REAL_DIRS:
    fs = files(d_)
    if fs:
        scores[d_] = p_fake(fs)
        print(f"{d_:22s} n={len(fs):3d} mean={scores[d_].mean():.3f}")

fake = np.concatenate([scores[d_] for d_ in FAKE_DIRS if d_ in scores])
real = np.concatenate([scores[d_] for d_ in REAL_DIRS if d_ in scores])
print(f"\nAll fakes n={len(fake)}, all reals n={len(real)}, AUC={auc(fake, real):.3f}")
print("\nthreshold  fakes_caught  reals_flagged  balanced_acc")
best = (0, 0.5)
for t in np.arange(0.30, 0.86, 0.05):
    rec = (fake > t).mean()
    fpr = (real > t).mean()
    ba = (rec + (1 - fpr)) / 2
    print(f"  {t:.2f}       {rec * 100:5.0f}%        {fpr * 100:5.0f}%        {ba:.3f}")
    if ba > best[0]:
        best = (ba, t)
print(f"\nbest threshold ~ {best[1]:.2f} (balanced accuracy {best[0]:.3f})")