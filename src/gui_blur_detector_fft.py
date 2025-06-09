import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import sys


def is_blur_fft(image, d=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    total_energy = magnitude.sum()
    h, w = magnitude.shape
    crow, ccol = h // 2, w // 2
    mask = np.ones((h, w), np.uint8)
    mask[crow - d : crow + d, ccol - d : ccol + d] = 0
    high_freq_energy = (magnitude * mask).sum()
    hf_ratio = high_freq_energy / (total_energy + 1e-8)

    logE = np.log(total_energy + 1e-8)

    # New thresholds
    if logE < 23.0:
        return False, total_energy, hf_ratio  # Not Blurred
    elif hf_ratio < 0.8815:
        return True, total_energy, hf_ratio  # Blurred
    else:
        return False, total_energy, hf_ratio  # Not Blurred


def main_gui_fft():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an Image (FFT‐based Blur Detector)",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")],
    )
    if not file_path:
        print("❌ No image selected. Exiting.")
        sys.exit(1)

    image = cv2.imread(file_path)
    if image is None:
        print("❌ Failed to load the image. Exiting.")
        sys.exit(1)

    # You can tweak these parameters if your images are much larger/smaller:
    D = 30
    TOTAL_ENERGY_TH = 1e7
    HF_RATIO_TH = 0.04

    is_blur, tot_e, hf_ratio = is_blur_fft(image)

    status = "Blurred" if is_blur else "Not Blurred"
    print("\n🔍 FFT‐Based Blur Detection Results:")
    print(f"Image: {status}")
    print(f"  Total Energy       = {tot_e:.2e}  (Threshold: {TOTAL_ENERGY_TH:.2e})")
    print(f"  High‐Freq Ratio    = {hf_ratio:.5f}  (Threshold: {HF_RATIO_TH:.5f})")

    # Display with overlay text
    display = image.copy()
    color = (0, 0, 255) if is_blur else (0, 255, 0)
    cv2.putText(
        display,
        f"{status} (R={hf_ratio:.3f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    cv2.imshow("FFT‐Based Blur Detector", display)
    print("\n📌 Press any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_gui_fft()
