import numpy as np
import cv2
from matplotlib import pyplot as plt


def compute_dense_depth_map(img1_path, img2_path, K, dist, b):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1 = cv2.undistort(img1, K, dist)
    img2 = cv2.undistort(img2, K, dist)

    # def crop_black(img, margin=30):
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    #     coords = cv2.findNonZero(thresh)
    #     if coords is None:
    #         return img, (0, 0, img.shape[1], img.shape[0])
    #     x, y, w, h = cv2.boundingRect(coords)
    #     x += margin; y += margin
    #     w -= 2*margin; h -= 2*margin
    #     return img[y:y+h, x:x+w], (x, y, w, h)

    # img1_c, roi = crop_black(img1)
    # x, y, w, h  = roi
    # img2_c       = img2[y:y+h, x:x+w]

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Améliorer le contraste avant SGBM
    gray1 = cv2.equalizeHist(gray1)
    gray2 = cv2.equalizeHist(gray2)

    stereo = cv2.StereoSGBM_create(
    # minDisparity=0,
    numDisparities=128,      # Ensure this covers the horizontal shift between your photos
    blockSize=5,
    # P1=8 * 3 * 9**2,  # Penalty for small changes (smoothness)
    # P2=32 * 3 * 9**2, # Penalty for large changes (discontinuity)
    # disp12MaxDiff=1,         # Stricter left-right check to reduce artifacts
    # uniquenessRatio=10,      # Higher value reduces "fake" matches in low-texture areas
    # speckleWindowSize=100,   # Filter out small noise clusters
    # speckleRange=32,         # Increased to merge fragmented depth regions
    # mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY # Better for performance/smoothness
)

    disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0
    disparity = cv2.medianBlur(disparity, 5)

    valid_mask = disparity > 1.0
    depth_vis  = np.zeros_like(disparity, dtype=np.uint8)

    if valid_mask.sum() > 0:
        # Utiliser percentiles pour etirer les couleurs
        p10 = np.percentile(disparity[valid_mask], 10)
        p90 = np.percentile(disparity[valid_mask], 90)
        print(f"Disparite p10={p10:.1f}  p90={p90:.1f}")

        clipped = np.clip(disparity, p10, p90)
        depth_vis[valid_mask] = (
            255 * (clipped[valid_mask] - p10) / (p90 - p10)
        ).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Image gauche', fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(disparity, cmap='plasma')
    axes[1].set_title('Carte de disparite', fontsize=13)
    axes[1].axis('off')

    axes[2].imshow(depth_vis, cmap='plasma', vmin=0, vmax=255)
    axes[2].set_title('Depth Map (jaune=proche, violet=loin)', fontsize=13)
    axes[2].axis('off')

    plt.suptitle('Dense Depth Map — StereoSGBM', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('depth_map_final.png', dpi=150, bbox_inches='tight')
    plt.show()

    return disparity, depth_vis