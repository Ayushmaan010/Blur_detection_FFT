import cv2
import numpy as np


def compute_fft_features(image, d=30):
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
    return hf_ratio, logE


def compute_laplacian_var(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var


def webcam_blur_detection_fft_temporal():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        return

    print("🎥 Webcam started. Press 'q' to quit.")
    prev_gray = None
    motion_var_threshold = 1000  # Lower means more sensitive to motion blur
    laplacian_threshold = 60  # Laplacian var below this → possibly blurred
    hf_ratio_threshold = 0.8815  # Your tuned FFT threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Computing the FFT high-frequency ratio
        hf_ratio, logE = compute_fft_features(frame, d=30)

        # Computing Laplacian variance
        lap_var = compute_laplacian_var(frame)

        # Computing the temporal difference if previous frame exists
        temporal_blur = False
        if prev_gray is not None:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, curr_gray)
            motion_var = diff.var()
            temporal_blur = motion_var < motion_var_threshold
            prev_gray = curr_gray
        else:
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Combining all the three blur indicators
        is_blur = (
            logE < 23.0
            or hf_ratio < hf_ratio_threshold
            or lap_var < laplacian_threshold
            or temporal_blur
        )

        # Displaying result
        status = "Blurred" if is_blur else "Not Blurred"
        color = (0, 0, 255) if is_blur else (0, 255, 0)
        overlay_text = f"{status} | R={hf_ratio:.3f} | Lap={lap_var:.1f} | T:{'YES' if temporal_blur else 'NO'}"

        cv2.putText(
            frame,
            overlay_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        cv2.imshow("Webcam Blur Detection (FFT + Laplacian + Temporal)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    webcam_blur_detection_fft_temporal()
