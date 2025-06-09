## Notes
Works best on 400x400 to 1000x1000 images
No training required; FFT handles signal-level features
Extendable to ML later by extracting these features as input (work in progress)

## Author
Developed by Ayushmaan Singh — interested in frequency domain, signal processing, and computer vision ml/dl.

## Sample Output (from Webcam)
Blurred | R=0.851 | Lap=34.2 | T:YES
Not Blurred | R=0.963 | Lap=94.6 | T:NO

## Technical Approach
While blur detection using Laplacian variance is a known computer vision technique, 
this project implements a novel FFT-based pipeline featuring:
- Frequency-domain energy analysis
- Optimized threshold calibration (logE < 23.0 and hf_ratio < 0.8815)
- Multi-mode operation (batch, GUI, webcam)
- Temporal analysis for motion blur detection

