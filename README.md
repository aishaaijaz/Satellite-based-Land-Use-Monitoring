# Satellite Change Detection using Classical Image Processing (OSCD)

A complete **classical computer vision pipeline** for detecting land‑use changes from satellite imagery using the **Onera Satellite Change Detection (OSCD) Dataset**.
This project focuses on interpretable image‑processing techniques instead of deep learning, making it lightweight, explainable, and easy to run on standard hardware.

---

# Project Overview

This system detects changes between **"Before"** and **"After"** satellite images by combining multiple visual cues:

* Intensity differences
* Colour vector differences
* Texture (gradient) variations

The outputs include:

* Binary change masks
* Red overlay visualizations
* Evaluation metrics (IoU, Precision, Recall, F1)
* Comprehensive performance plots and pipeline visualizations

---

# Features

Classical Image Processing Based (No GPU Required)
Multi‑Strategy Change Detection
Voting‑Based Mask Fusion
Morphological Noise Removal
Region‑wise Evaluation Metrics
Automated Sensitivity Tuning
Advanced Visualization Suite

---

# Methodology

## Data Loading

Satellite image pairs and ground‑truth masks are loaded region‑wise from the OSCD dataset structure.

## Preprocessing

The following operations are applied implicitly during detection:

* RGB → Grayscale conversion
* Gaussian smoothing (noise reduction)
* Gradient computation (texture extraction)
* Normalization of difference maps

Images are processed at **original OSCD resolution** — no resizing is performed during detection.

## Change Detection Strategies

Three independent change maps are computed:

### • Intensity Change

Absolute grayscale difference with adaptive thresholding.

### • Colour Change

Euclidean RGB vector distance between before/after images.

### • Texture Change

Gradient magnitude difference using Sobel filters.

## Voting‑Based Fusion

The three binary masks are combined using a configurable voting system:

* Conservative → 3/3 agreement required
* Balanced → 2/3 agreement required
* Sensitive → detects more subtle changes

## Post‑Processing

* Morphological Opening → removes small noise
* Morphological Closing → fills holes in detected regions
* Connected Component Filtering → removes regions below minimum area threshold

## Evaluation

For regions with ground truth:

* IoU (Intersection over Union)
* Precision
* Recall
* F1 Score

## Visualization

The project automatically generates:

* Multi‑region result grids
* Pipeline breakdown visualizations
* Best vs Worst region comparisons
* Statistical performance plots

---

# Dataset

**Onera Satellite Change Detection (OSCD)**

* Spatial Resolution: ~50 cm per pixel
* Image Sizes: Variable (e.g., 600×600 to 1024×1024)
* Regions: 24 geographically diverse locations

Dataset Structure:

```
region_name/
 ├── pair/
 │   ├── before.png
 │   └── after.png
 └── cm/
     └── ground_truth.png
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/oscd-change-detection.git
cd oscd-change-detection
```

Install dependencies:

```bash
pip install opencv-python numpy matplotlib scikit-learn scikit-image
```

---

# Usage

Update dataset paths inside the script:

```python
IMAGES_ROOT = "path_to_oscd_images"
LABELS_ROOT = "path_to_oscd_labels"
```

Run:

```bash
python main.py
```

The pipeline will:

1. Tune optimal sensitivity
2. Evaluate all regions
3. Generate visualizations automatically

---

# Example Results

Average Performance (Balanced Sensitivity):

* IoU ≈ 0.13
* Precision ≈ 0.18
* Recall ≈ 0.44
* F1 Score ≈ 0.23

The system performs strongest on dense urban regions where structural changes are prominent.

---

# Generated Outputs

The following visualizations are saved automatically:

```
multi_region_grid_*.png
best_worst_comparison_*.png
performance_statistics.png
[region]_pipeline_*.png
```

---

# Technologies Used

* Python
* OpenCV
* NumPy
* Matplotlib
* Scikit‑Image (SSIM)
* Scikit‑Learn (Metrics)

---

# Future Improvements

* Deep Learning Baseline Comparison
* Superpixel‑Based Change Detection
* Temporal Satellite Sequence Analysis
* Lightweight Web Interface

---

# Author

**Aisha Aijaz**
Computer Science Student | AI & ML Enthusiast

**Nishtha Priya**
Computer Science Student | AI & ML Enthusiast
(https://github.com/Nishtha-Priya)

---

# Acknowledgements

* Onera Satellite Change Detection Dataset
* OpenCV & Scikit‑Image Communities

---

# License

This project is for educational and research purposes.
