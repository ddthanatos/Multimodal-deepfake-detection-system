import argparse, io, os, random, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T
from PIL import Image, ImageFilter
import pipeline as p

DEV = p.device


class Degrade:
    """Randomly destroy resolution / compression shortcuts."""
    def __call__(self, img):
        if random.random() < 0.5:
            s = random.choice([64, 96, 128, 160, 200])
            img = img.resize((s, s), Image.BILINEAR).resize((256, 256), Image.BILINEAR)
        if random.random() < 0.7:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=random.randint(30, 95))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5)))
        return img


NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = T.Compose([
    T.Resize((256, 256)),
    Degrade(),
    T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(),
    NORM,
])
eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), NORM])


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    ranks = np.empty(len(s))
    ranks[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1 = (y == 1).sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def subset(ds, n):
    if n and n < len(ds):
        return Subset(ds, random.sample(range(len(ds)), n))
    return ds


def evaluate(model, loader):
    model.eval()
    ys, ss = [], []
    with torch.no_grad():
        for x, y in loader:
            pr = torch.softmax(model(x.to(DEV)), 1)[:, 1].cpu()
            ys += y.tolist()
            ss += pr.tolist()
    ys, ss = np.array(ys), np.array(ss)
    return float(((ss > 0.5) == (ys == 1)).mean()), auc(ys, ss)


def main(a):
    print("device:", DEV)
    tr = datasets.ImageFolder(os.path.join(a.data, "train"), train_tf)
    va = datasets.ImageFolder(os.path.join(a.data, "valid"), eval_tf)
    assert tr.class_to_idx == {"fake": 0, "real": 1}, tr.class_to_idx

    train_set = subset(tr, a.limit)
    val_n = (a.limit // 5) if a.limit else 5000
    val_set = subset(va, val_n)

    trl = DataLoader(train_set, batch_size=a.bs, shuffle=True,
                     num_workers=a.workers, pin_memory=True, drop_last=True)
    val = DataLoader(val_set, batch_size=a.bs, num_workers=a.workers)

    model = p.DeepfakeDetector().to(DEV)
    src = os.path.join(p.MODEL_DIR, "efficientnet_b4_best.pth")
    if os.path.exists(src) and not a.scratch:
        ck = torch.load(src, map_location=DEV, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        print("Starting from", src)

    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": a.lr},
        {"params": model.classifier.parameters(), "lr": a.lr * 5},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    use_amp = DEV.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = 0.0
    out = os.path.join(p.MODEL_DIR, "efficientnet_b4_ft.pth")

    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        for i, (x, y) in enumerate(trl):
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            with torch.autocast(device_type=DEV.type, enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if i % 50 == 0:
                print(f"ep {ep} step {i}/{len(trl)} loss {loss.item():.4f}", flush=True)
        sched.step()
        acc, a_auc = evaluate(model, val)
        print(f"epoch {ep}: val_acc {acc:.4f} val_auc {a_auc:.4f} ({time.time()-t0:.0f}s)")
        if a_auc >= best:
            best = a_auc
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "val_acc": acc, "val_auc": a_auc}, out)
            print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="faces/real_vs_fake/real-vs-fake")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="train on a random subset of N images")
    ap.add_argument("--scratch", action="store_true")
    main(ap.parse_args())