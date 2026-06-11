import os
import shutil


ORIGINAL_DIR = "/Users/pradeepkd/Desktop/projectphase1/archive/colored_images"
SYNTHETIC_DIR = "/Users/pradeepkd/Desktop/projectphase1/synthetic_images"
MERGED_DIR = "/Users/pradeepkd/Desktop/projectphase1/dataset_merged"


CLASSES = ["Proliferate_DR", "Severe", "Moderate", "Mild", "No_DR"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def copy_all(src_dir: str, dst_dir: str) -> None:
    if not os.path.isdir(src_dir):
        return
    for root, _, files in os.walk(src_dir):
        for f in files:
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            src_path = os.path.join(root, f)
            dst_path = os.path.join(dst_dir, f)
            # If filename exists, uniquify
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(f)
                i = 1
                while os.path.exists(os.path.join(dst_dir, f"{base}_{i}{ext}")):
                    i += 1
                dst_path = os.path.join(dst_dir, f"{base}_{i}{ext}")
            shutil.copy2(src_path, dst_path)


def main() -> None:
    ensure_dir(MERGED_DIR)
    for cls in CLASSES:
        merged_class_dir = os.path.join(MERGED_DIR, cls)
        ensure_dir(merged_class_dir)

        # Copy originals
        original_class_dir = os.path.join(ORIGINAL_DIR, cls)
        copy_all(original_class_dir, merged_class_dir)

        # Copy synthetic
        synthetic_class_dir = os.path.join(SYNTHETIC_DIR, cls)
        copy_all(synthetic_class_dir, merged_class_dir)

        print(f"[OK] Merged class {cls}")

    print(f"[DONE] Merged dataset at: {MERGED_DIR}")


if __name__ == "__main__":
    main()


