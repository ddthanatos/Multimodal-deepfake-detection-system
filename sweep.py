import os
import numpy as np
import torch
import cv2
from PIL import Image
import pipeline as p

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
REAL_KEYS = ("real", "lady")
FAKE_KEYS = ("fake", "gan", "synthesis")
FRAMES_PER_VIDEO = 6

model = p._load_image_model()
ck = torch.load(os.path.join(p.MODEL_DIR, "efficientnet_b4_best.pth"),
                map_location="cpu", weights_only=False)
print("Checkpoint info:", {k: v for k, v in ck.items() if k != "model_state_dict"})


def label_of(path):
    s = path.lower()
    if any(k in s for k in FAKE_KEYS):
        return 1
    if any(k in s for k in REAL_KEYS):
        return 0
    return None


# ---- collect samples: (name, label, rgb) ----
samples = []
for root, exts in (("images", IMG_EXT), ("videos", VID_EXT)):
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if not fn.lower().endswith(exts):
                continue
            path = os.path.join(dp, fn)
            lab = label_of(path)
            if lab is None:
                continue
            if exts == IMG_EXT:
                try:
                    samples.append((path, lab, np.array(Image.open(path).convert("RGB"))))
                except Exception:
                    print("skip (unreadable):", path)
            else:
                cap = cv2.VideoCapture(path)
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for i in np.linspace(0, max(n - 1, 0), FRAMES_PER_VIDEO, dtype=int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                    ok, fr = cap.read()
                    if ok:
                        samples.append((path, lab, cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
                cap.release()

names = sorted({s[0] for s in samples})
print(f"{len(samples)} samples from {len(names)} files:")
for n in names:
    print("  ", "FAKE" if label_of(n) else "REAL", n)
if not any(s[1] == 1 for s in samples) or not any(s[1] == 0 for s in samples):
    raise SystemExit("Need at least one real and one fake sample.")

# ---- face crops (cached) ----
CROPS = {"none": None, "m0.0": 0.0, "m0.2": 0.2, "m0.4": 0.4}
crop_cache = {}
for key, margin in CROPS.items():
    for i, (_, _, rgb) in enumerate(samples):
        if margin is None:
            crop_cache[(key, i)] = rgb
        else:
            p.FACE_MARGIN = margin
            c = p._crop_face(rgb)
            crop_cache[(key, i)] = c if c is not None and c.size > 0 else rgb

NORMS = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "half": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    "none": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
}
SIZES = [224, 256, 299, 380]
ORDERS = ["RGB", "BGR"]


def tensorize(rgb, size, norm, order):
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    if order == "BGR":
        img = img[:, :, ::-1]
    mean, std = NORMS[norm]
    img = (img - np.array(mean, np.float32)) / np.array(std, np.float32)
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


results = []
total = len(CROPS) * len(SIZES) * len(NORMS) * len(ORDERS)
done = 0
for crop_key in CROPS:
    for size in SIZES:
        for norm in NORMS:
            for order in ORDERS:
                done += 1
                p1 = []
                for start in range(0, len(samples), 16):
                    idxs = range(start, min(start + 16, len(samples)))
                    batch = torch.stack([tensorize(crop_cache[(crop_key, i)], size, norm, order)
                                         for i in idxs]).to(p.device)
                    with torch.no_grad():
                        p1 += torch.softmax(model(batch), 1)[:, 1].cpu().tolist()
                # average per file first, then per class
                per_file = {}
                for (name, lab, _), v in zip(samples, p1):
                    per_file.setdefault(name, (lab, []))[1].append(v)
                file_means = {n: (l, float(np.mean(v))) for n, (l, v) in per_file.items()}
                real = [m for l, m in file_means.values() if l == 0]
                fake = [m for l, m in file_means.values() if l == 1]
                gap = float(np.mean(real) - np.mean(fake))   # >0 means p1 = REAL
                results.append((abs(gap), gap, crop_key, size, norm, order, file_means))
                print(f"[{done}/{total}] crop={crop_key} size={size} norm={norm} {order} gap={gap:+.3f}", flush=True)

results.sort(key=lambda r: -r[0])
print("\n===== TOP 10 VARIANTS (largest real-vs-fake separation) =====")
for _, gap, crop_key, size, norm, order, fm in results[:10]:
    meaning = "p1=REAL" if gap > 0 else "p1=FAKE"
    print(f"\ncrop={crop_key} size={size} norm={norm} order={order}  gap={gap:+.3f} ({meaning})")
    for n, (l, m) in sorted(fm.items()):
        print(f"    {'FAKE' if l else 'REAL'}  p1={m:.3f}  {n}")

print("\n===== CURRENT PIPELINE SETTING (crop=m0.2 size=224 norm=imagenet RGB) =====")
for r in results:
    if r[2:6] == ("m0.2", 224, "imagenet", "RGB"):
        print(f"gap={r[1]:+.3f}")
        for n, (l, m) in sorted(r[6].items()):
            print(f"    {'FAKE' if l else 'REAL'}  p1={m:.3f}  {n}")