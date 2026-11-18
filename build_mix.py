import argparse, glob, os, random, shutil

IMG = (".jpg", ".jpeg", ".png", ".webp")


def files(d):
    return [f for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)
            if f.lower().endswith(IMG)]


def put(fs, dst, tag):
    os.makedirs(dst, exist_ok=True)
    for i, f in enumerate(fs):
        shutil.copy(f, os.path.join(dst, f"{tag}_{i}{os.path.splitext(f)[1].lower()}"))


def main(a):
    random.seed(0)
    for split, frac in (("train", 0.9), ("valid", 0.1)):
        for cls in ("fake", "real"):
            base = files(os.path.join(a.base, split, cls))
            base = random.sample(base, min(a.n_base if split == "train" else a.n_base // 10, len(base)))
            put(base, os.path.join(a.out, split, cls), "base")

    extra_fake = files(a.extra_fake)
    random.shuffle(extra_fake)
    cut = int(len(extra_fake) * 0.9)
    parts = {"train": extra_fake[:cut], "valid": extra_fake[cut:]}
    for split, fs in parts.items():
        for r in range(a.repeat if split == "train" else 1):
            put(fs, os.path.join(a.out, split, "fake"), f"extra{r}")

    if a.extra_real and os.path.isdir(a.extra_real):
        extra_real = files(a.extra_real)
        random.shuffle(extra_real)
        cut = int(len(extra_real) * 0.9)
        for split, fs in (("train", extra_real[:cut]), ("valid", extra_real[cut:])):
            for r in range(a.repeat if split == "train" else 1):
                put(fs, os.path.join(a.out, split, "real"), f"xreal{r}")

    for split in ("train", "valid"):
        for cls in ("fake", "real"):
            n = len(files(os.path.join(a.out, split, cls)))
            print(f"{split}/{cls}: {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="faces/real_vs_fake/real-vs-fake")
    ap.add_argument("--extra_fake", default="extra/fake")
    ap.add_argument("--extra_real", default="extra/real")
    ap.add_argument("--out", default="mix")
    ap.add_argument("--n_base", type=int, default=8000, help="base images per class in train")
    ap.add_argument("--repeat", type=int, default=2, help="copies of extra images (augmentation makes them differ)")
    main(ap.parse_args())