import random
import shutil
from pathlib import Path

def split_dataset(src_root, dst_root,train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)

    src_root = Path(src_root)
    dst_root = Path(dst_root)

    #Create destination root directory if it doesn't exist
    dst_root.mkdir(parents=True, exist_ok=True)

    img_dir = src_root / "images"
    lbl_dir = src_root / "labels"

    # Create destination directories with checks
    for split in ["train", "val", "test"]:
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # .glob to get all image files
    images = list(img_dir.glob("*.jpg"))

    # Only keep images with a corresponding label
    images = [p for p in images if (lbl_dir / f"{p.stem}.txt").exists()]
    random.shuffle(images)

    n = len(images)
    n_train = int(train_ratio * n)
    n_val   = int(val_ratio * n)

    train_imgs = images[:n_train]
    val_imgs   = images[n_train:n_train+n_val]
    test_imgs  = images[n_train+n_val:]

    def copy_pair(img_path, split):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        shutil.copy2(img_path, dst_root / "images" / split / img_path.name)
        shutil.copy2(lbl_path, dst_root / "labels" / split / f"{img_path.stem}.txt")

    for p in train_imgs: copy_pair(p, "train")
    for p in val_imgs:   copy_pair(p, "val")
    for p in test_imgs:  copy_pair(p, "test")

    #To notify that the split is done
    print(f"Split of dataset is completed to - {dst_root}")
    #To check if the split is correct
    print(f"Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)} (Total={n})")

if __name__ == "__main__":
    split_dataset("Dataset", "Split_dataset")
