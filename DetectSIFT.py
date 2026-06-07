import cv2
import numpy as np 
import matplotlib.pyplot as plt 


def remove_curve(path_img, k, dist):
    img = cv2.imread(path_img)
    img_fixed = cv2.undistort(img, k, dist)
    return img_fixed


def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def SIFT_Match(img_left, img_right, k, dist, path_folder, lowe_ratio: float = 0.80, show_plot: bool = True):
    img1 = remove_curve(img_left, k, dist)
    img2 = remove_curve(img_right, k, dist)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    sift = cv2.SIFT_create()

    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    print(f"Keypoints detected — Left: {len(kp1)}, Right: {len(kp2)}")

    bfMatch = cv2.BFMatcher()
    matches = bfMatch.knnMatch(desc1, desc2, k=2)

    goodMatch = []
    for m, n in matches:
        if m.distance < lowe_ratio * n.distance:
            goodMatch.append(m)
    print(f"After Lowe ratio test : {len(goodMatch)} matches")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in goodMatch])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in goodMatch])

     # Essential matrix with RANSAC
    E, mask_E = cv2.findEssentialMat(
        pts1, pts2,
        cameraMatrix=k,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    # Use ONLY the Essential mat mask for filtering
    mask_bool = mask_E.ravel().astype(bool)
    pts1_in   = pts1[mask_bool]
    pts2_in   = pts2[mask_bool]
    good      = [m for m, keep in zip(goodMatch, mask_bool) if keep]

    print(f"After RANSAC epipolar filter : {len(good)} matches")

    # recoverPose separately — just for info, don't use its mask for filtering
    _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, cameraMatrix=k)
    print(f"Translation vector t (direction): {t.ravel()}")

    import os
    os.makedirs(path_folder, exist_ok=True)
    np.savetxt(f"{path_folder}/x1_positions.txt", pts1_in[:, 0])
    np.savetxt(f"{path_folder}/y1_positions.txt", pts1_in[:, 1])
    np.savetxt(f"{path_folder}/x2_positions.txt", pts2_in[:, 0])
    np.savetxt(f"{path_folder}/y2_positions.txt", pts2_in[:, 1])

    if show_plot:
        img_match = cv2.drawMatches(
            img1, kp1, img2, kp2, good, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB))
        plt.title(f"SIFT matches after epipolar filtering – {len(good)} points")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return good, R, t