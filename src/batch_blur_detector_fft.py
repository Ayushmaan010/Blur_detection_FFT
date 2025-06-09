import cv2
import numpy as np
import os
from math import log


def is_blur_fft(image, d: int = 30) -> tuple[bool, float, float]:
    """
    FFT-based blur detector.

    Returns:
        is_blur      – True  → Blurred
                       False → Not Blurred
        total_energy – Sum of magnitude spectrum
        hf_ratio     – High-frequency energy ratio
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Full 2-D FFT
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(fshift)

    # Total spectral energy
    total_energy = magnitude.sum()

    # Mask out low frequencies (central (2d × 2d) square)
    h, w = magnitude.shape
    crow, ccol = h // 2, w // 2
    mask = np.ones((h, w), np.uint8)
    mask[crow - d : crow + d, ccol - d : ccol + d] = 0
    high_freq_energy = (magnitude * mask).sum()

    hf_ratio = high_freq_energy / (total_energy + 1e-8)
    logE = log(total_energy + 1e-8)

    # Tuned thresholds (from your grid-search)
    if logE < 23.0:  # Uniform / low-detail sharp
        return False, total_energy, hf_ratio
    if hf_ratio < 0.8815:  # Texture present but high-freq suppressed
        return True, total_energy, hf_ratio
    return False, total_energy, hf_ratio


def scan_single_folder(folder_path: str, label: str) -> None:
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)]
    )
    if not files:
        print(f"⚠️  No images in {folder_path}")
        return

    print(f"\n─── {label.upper()} images ({len(files)}) ─────────────────────────────")
    print(f"{'Filename':25} {'Status':12} {'logE':>10} {'HF_ratio':>10}")
    print("-" * 60)

    for fname in files:
        img = cv2.imread(os.path.join(folder_path, fname))
        if img is None:
            print(f"⚠️  Skipping invalid image: {fname}")
            continue

        blurred, tot_e, hf_ratio = is_blur_fft(img)
        status = "Blurred" if blurred else "Not Blurred"
        print(f"{fname:25} {status:12} {log(tot_e):10.2f} {hf_ratio:10.5f}")


def batch_blur_detection_fft(root_images: str) -> None:
    if not os.path.isdir(root_images):
        print("❌ Supplied path is not a directory.")
        return

    print(f"\n📂 Root images folder: {root_images}")

    # Look for sub-folders
    sharp_dir = os.path.join(root_images, "sharp")
    blur_dir = os.path.join(root_images, "blur")

    any_scanned = False
    if os.path.isdir(sharp_dir):
        scan_single_folder(sharp_dir, "sharp")
        any_scanned = True
    if os.path.isdir(blur_dir):
        scan_single_folder(blur_dir, "blur")
        any_scanned = True

    if not any_scanned:  # Fallback: scan root itself
        print("⚠️  No 'sharp/' or 'blur/' sub-folders found – scanning root folder.")
        scan_single_folder(root_images, "root")


if __name__ == "__main__":
    folder = input("📁 Enter *root* images folder path: ").strip()
    batch_blur_detection_fft(folder)
