# 📷 Stereo Vision — 3D Reconstruction Mini Project

## Overview

This project reconstructs a **3D scene** from a pair of stereo images taken with a single camera. Three product boxes (Kayali Perfume, Derma-T, and Touché Serum) were placed at different depths in front of the camera. By introducing a small **horizontal baseline shift** between two shots, the setup simulates a stereo camera rig. The pipeline goes from camera calibration all the way to a dense depth map and a 3D point cloud with fitted bounding boxes.

---

## Scene Setup : How the Images Were Taken

| | |
|---|---|
| **Objects** | 3 boxes placed at increasing distances from the camera |
| **Kayali Perfume** | ~400 mm from the camera |
| **Derma-T** | ~505 mm from the camera |
| **Touché Serum** | ~565 mm from the camera |
| **Baseline (b)** | 60 mm horizontal shift between left and right shots |
| **Method** | Single camera moved slightly to the right to simulate a stereo pair |

The left image (`imageL.jpg`) and right image (`imageR.jpg`) share the same scene but are captured from positions separated by **60 mm** along the horizontal axis. This baseline is what allows depth to be inferred from the disparity (pixel offset) between matched points.

---

## Pipeline Overview

```
Camera Calibration (Zhang's method)
          │
          ▼
   SIFT Feature Detection & Matching
          │
          ▼
   Epipolar Filtering (Essential Matrix + RANSAC)
          │
          ▼
   3D Reconstruction (Triangulation)
          │
          ▼
   Depth Map (StereoSGBM)
          │
          ▼
   Visualization (Point Cloud + Bounding Boxes)
```

---

## Step 1 : Camera Calibration (`CalibrerCamera.py`)

**Method: Zhang's planar checkerboard method (Zhengyou Zhang, 2000)**

A set of checkerboard images was captured from multiple angles and distances using a mobile phone. The calibration board used a **7×9 inner corner** grid.

### Process

The calibration images were first **downscaled by 50%** to reduce processing time while preserving corner detection accuracy.

For each image:
- The checkerboard corners are detected with `cv2.findChessboardCorners`
- Detected corners are refined to sub-pixel accuracy using `cv2.cornerSubPix`
- The refined corners are stored as 2D image points (`imgpoints`)
- Corresponding 3D world points (`objpoints`) are defined on a flat plane (Z=0)

Once all images are processed, `cv2.calibrateCamera` computes:

| Parameter | Description |
|---|---|
| **K (mtx)** | Intrinsic camera matrix (focal lengths fx, fy and principal point cx, cy) |
| **dist** | Lens distortion coefficients (radial + tangential) |
| **rvecs** | Rotation vectors for each calibration image |
| **tvecs** | Translation vectors for each calibration image |

All parameters are saved as `.npy` files in `camera_parameters/` for use in subsequent steps.

### Reprojection Error

The calibration quality is verified by reprojecting the 3D object points back onto the image plane and measuring the pixel-level error. A low reprojection error (ideally < 1.0 px) indicates a reliable calibration.

---

## Step 2 : SIFT Feature Detection & Matching (`DetectSIFT.py`)

### Distortion Correction

Before any processing, both images are **undistorted** using the calibration parameters (K, dist) via `cv2.undistort`, removing the lens barrel/pincushion distortion that would otherwise corrupt the depth estimates.

### SIFT Detection

**SIFT (Scale-Invariant Feature Transform)** is applied to both grayscale images to detect keypoints and compute 128-dimensional descriptors at each. SIFT features are robust to changes in scale, rotation, and illumination.

### Feature Matching — BFMatcher + Lowe's Ratio Test

A **Brute-Force Matcher (BFMatcher)** performs k-NN matching (k=2) between descriptors from the left and right images. **Lowe's ratio test** (threshold = 0.75) filters ambiguous matches:

```
if m.distance < 0.75 × n.distance → keep match
```

### Epipolar Filtering — Essential Matrix + RANSAC

Good matches are further filtered using the **Essential Matrix**, which encodes the geometric relationship between the two camera positions. Matches that do not satisfy the **epipolar constraint** (i.e. the matched point in the right image must lie on the epipolar line corresponding to the left image point) are rejected via **RANSAC**.

This gives a clean, geometrically consistent set of correspondences.

### Camera Pose Recovery

From the Essential Matrix, `cv2.recoverPose` extracts the **rotation matrix R** and **translation vector t** between the two camera positions, confirming the direction and magnitude of the baseline shift.

Matched point coordinates are saved to text files in `positions/` for the reconstruction step.

---

## Step 3 — 3D Reconstruction (`Stereo.py`)

### Depth from Disparity

For each matched point pair (u_L, v_L) ↔ (u_R, v_R), the **depth Z** is computed using the stereo triangulation formula:

```
Z = (b × fx) / |u_L − u_R|
```

Where:
- `b` = baseline = 60 mm
- `fx` = horizontal focal length (from K)
- `|u_L − u_R|` = horizontal disparity between matched points

Points with disparity < 1 pixel or depth > 5000 mm are rejected as unreliable.

The **3D coordinates (X, Y, Z)** are then recovered using the standard pinhole camera back-projection:

```
X = (u_L − cx) × Z / fx
Y = (v_L − cy) × Z / fy
```

### Output

A 3D point cloud is returned as an array of (X, Y, Z) coordinates in millimeters.

---

## Step 4 : Dense Depth Map (`depth.py`)

In addition to the sparse SIFT-based reconstruction, a **dense depth map** is computed using `cv2.StereoSGBM` (Semi-Global Block Matching), which estimates disparity at every pixel — not just at matched keypoints.

### Process

Both images are undistorted, then histogram-equalized to improve contrast before SGBM matching. Key parameters:

| Parameter | Value | Role |
|---|---|---|
| `numDisparities` | 128 | Maximum disparity search range |
| `blockSize` | 5 | Matching window size |
| Post-processing | Median blur (5×5) | Removes salt-and-pepper noise |

The disparity map is converted to a visual depth map by clipping to the 10th–90th percentile range and mapping to a color scale (plasma colormap: **yellow = close, purple = far**).

The result is saved as `depth_map_final.png`.

---

## Step 5 : Visualization (`Stereo.py` + `Main.py`)

Three visualizations are produced:

### 5a : Raw 3D Point Cloud
A scatter plot of all reconstructed 3D points, colored by depth (Z), rendered in 3D using matplotlib.

### 5b : Clustered Bounding Boxes
**K-Means clustering** (k=3) groups the 3D points by depth into three clusters corresponding to the three boxes. For each cluster, a bounding box is drawn using the **known physical dimensions** of each object:

| Object | Width (mm) | Height (mm) | Depth (mm) |
|---|---|---|---|
| Kayali Perfume | 40 | 72 | 138 |
| Derma-T | 36 | 42 | 107 |
| Touché Serum | 31 | 31 | 112 |

Clusters are sorted by their mean Z value (nearest first) and matched to objects accordingly.

### 5c : Dense Depth Map
The StereoSGBM disparity and depth map displayed side-by-side with the left image.

---

## Step 6 : Validation (`validate2.py`)

The reconstructed Z values for each cluster are compared against the **ground-truth distances** (400, 505, 565 mm) to quantify the accuracy of the reconstruction. This gives a direct measure of how well the pipeline performs.

---

## File Structure

```
project/
├── imageL.jpg                    # Left stereo image
├── imageR.jpg                    # Right stereo image
├── mobile images/                # Checkerboard calibration photos
├── resized/                      # Downscaled calibration images
├── camera_parameters/
│   ├── mtx.npy                   # Intrinsic matrix K
│   ├── dist.npy                  # Distortion coefficients
│   ├── rvecs.npy                 # Rotation vectors
│   └── tvecs.npy                 # Translation vectors
├── positions/
│   ├── x1_positions.txt          # Left image matched x-coordinates
│   ├── y1_positions.txt          # Left image matched y-coordinates
│   ├── x2_positions.txt          # Right image matched x-coordinates
│   └── y2_positions.txt          # Right image matched y-coordinates
├── CalibrerCamera.py             # Step 1 — Camera calibration
├── DetectSIFT.py                 # Step 2 — Feature detection & matching
├── Stereo.py                     # Step 3 — 3D reconstruction & visualization
├── depth.py                      # Step 4 — Dense depth map (StereoSGBM)
├── Main.py                       # Entry point — runs the full pipeline
├── depth_map_final.png           # Output dense depth map
└── results_ex1/                  # Checkerboard corner visualizations
```

---

## How to Run

```bash
# 1. Install dependencies
pip install opencv-python numpy matplotlib scikit-learn scipy

# 2. Run camera calibration (only needed once)
python CalibrerCamera.py

# 3. Run the full stereo pipeline
python Main.py
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `opencv-python` | Calibration, SIFT, SGBM, undistortion |
| `numpy` | Matrix and array operations |
| `matplotlib` | 2D and 3D visualization |
| `scikit-learn` | K-Means clustering |
| `scipy` | Convex hull for mesh visualization |

---

## References

- Zhang, Z. (2000). *A Flexible New Technique for Camera Calibration*. IEEE TPAMI.
- Lowe, D.G. (2004). *Distinctive Image Features from Scale-Invariant Keypoints*. IJCV.
- Hirschmüller, H. (2008). *Stereo Processing by Semiglobal Matching and Mutual Information*. IEEE TPAMI.
- Fischler & Bolles (1981). *Random Sample Consensus (RANSAC)*. Communications of the ACM.
