import os
import numpy as np
import torch
import cv2
from PIL import Image
import pipeline as p

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")

img_model = p._load_image_model()
vid_model = p._load_video_model()


def list_files(root, exts):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if fn.lower().endswith(exts):
                out.append(os.path.join(dp, fn))
    return out


def raw(model, rgb, crop):
    p.USE_FACE_CROP = crop
    t, found = p._prep(rgb)
    with torch.no_grad():
        pr = torch.softmax(model(t.unsqueeze(0).to(p.device)), 1)[0].tolist()
    return [round(x, 3) for x in pr], found


print("\n=== IMAGES (raw softmax [p0, p1]) ===")
for f in list_files("images", IMG_EXT):
    try:
        rgb = np.array(Image.open(f).convert("RGB"))
    except Exception as e:
        print(f"{f}: could not open ({e})")
        continue
    full, _ = raw(img_model, rgb, False)
    crop, found = raw(img_model, rgb, True)
    print(f"{f:45s} full={full}  face_crop={crop}  face_found={found}")

print("\n=== VIDEOS (mean raw softmax over 10 frames) ===")
for f in list_files("videos", VID_EXT):
    cap = cv2.VideoCapture(f)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    res = {False: [], True: []}
    for i in np.linspace(0, max(n - 1, 0), 10, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        for crop in (False, True):
            res[crop].append(raw(vid_model, rgb, crop)[0])
    cap.release()
    if not res[False]:
        print(f"{f}: no frames read")
        continue
    print(f"{f:45s} full={np.mean(res[False], 0).round(3).tolist()}  "
          f"face_crop={np.mean(res[True], 0).round(3).tolist()}")

print("\nHow to read this: known fakes (celeb_synthesis.mp4, images_fake_image.jpg, files in images/gan) "
      "vs known reals (real-1.mp4, images_lady.jpg).\n"
      "Fakes high in p0 and reals high in p1 -> labels are correct (current default).\n"
      "Fakes high in p1 and reals high in p0 -> labels are flipped: set IMAGE_REAL_IDX=0 and VIDEO_REAL_IDX=0.")