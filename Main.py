import DetectSIFT as s 
import Stereo as ss
import numpy as np 
import cv2 
import validate2
import visualize_better
import depth as dp


BOX_SPECS = {
    "Kayali Perfume": {"dims": (40,  72,  138), "true_z": 400, "color": "#c0474a"},
    "Derma-T":        {"dims": (36,  42,  107), "true_z": 505, "color": "#1D9E75"},
    "Touché Serum":   {"dims": (31,  31,  112), "true_z": 565, "color": "#378ADD"},
}

# first step which is claibrating the camera is done 
k = np.load("camera_parameters/mtx.npy")
dist = np.load("camera_parameters/dist.npy")

print("Camera matrix K:")
print(k)
print(f"\nfx={k[0,0]:.1f}  fy={k[1,1]:.1f}  cx={k[0,2]:.1f}  cy={k[1,2]:.1f}\n")

# second step which is detecting sift 
goodMatches = s.SIFT_Match(
    "imageL.jpg", 
    "imageR.jpg", 
    k, dist, 
    "positions",
    lowe_ratio=0.75,
    show_plot=True
)

# third step which it recondtruction 3D 
b = 60.0
points = ss.reconstruct_3d("positions", b)
print(f"{len(points)} points 3D reconstructed")

# validate using actual depth 
if len(points) > 0:
    z_vals = points[:, 2]
    print(f"\nDepth statistics (Z in mm):")
    print(f"  Min  : {z_vals.min():.1f}")
    print(f"  Max  : {z_vals.max():.1f}")
    print(f"  Mean : {z_vals.mean():.1f}")
    print(f"  Std  : {z_vals.std():.1f}")
    print("\n  Expected peaks near: 300, 380, 480 mm")
    print("  If numbers are very different, re-measure baseline b carefully.\n")
 
# step 5 visualize data 
ss.visualize_3d(points)
ss.visualize_boxes_from_known_dimensions(points)
#ss.visualize_mesh_open3d(points) 

visualize_better.visualize_3d_boxes(
    points_3d      = points,
    box_specs      = BOX_SPECS,
    k              = 3,
    show_points    = True,
    show_ghost_boxes = True,   # draws true-size wireframe boxes at reconstructed Z
)


distance = [400, 505, 565] # Your distances in mmz
validate2.validate_reconstruction(points, distance)

disparity, depth_vis = dp.compute_dense_depth_map(
    "imageL.jpg",
    "imageR.jpg",
    k, dist, b=60.0
)