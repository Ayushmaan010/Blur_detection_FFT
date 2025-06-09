import cv2
import numpy as np
import os


def compute_freq_metrics(path, d=30):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 2D FFT and shift
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    crow, ccol = h // 2, w // 2

    total_energy = magnitude.sum()

    # Build high-frequency mask
    mask = np.ones((h, w), np.uint8)
    mask[crow - d : crow + d, ccol - d : ccol + d] = 0
    high_freq_energy = (magnitude * mask).sum()

    high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
    return total_energy, high_freq_ratio


# Change d if your images are bigger/smaller.
D = 30

for category in ("sharp", "blur"):
    print(f"\n--- {category.upper()} images ---")
    folder = os.path.join("images", category)
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        full = os.path.join(folder, fname)
        tot_e, hf_ratio = compute_freq_metrics(full, d=D)
        print(
            f"{fname:15} → total_energy={tot_e:12.2e},  high_freq_ratio={hf_ratio:.5f}"
        )
