#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob

def resize_images(input_folder, output_folder, scale=0.5):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            if img is None:
                continue
            h1,w1 = img.shape[:2]
            pixelsBefore = h1 * w1 
            resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            h2,w2 = resized.shape[:2]
            pixelsAfter = h2 * w2
            print(f"{filename}")
            print(f"Avant resize : {w1} x {h1} = {pixelsBefore} pixels")
            print(f"Après resize : {w2} x {h2} = {pixelsAfter} pixels\n")
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, resized)

            print(f"Resized: {filename}")

    print("All images resized.")

# Defining the dimensions of checkerboard

CHECKERBOARD = (7,9)            
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 
error =[]


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

nx = 7
#Enter the number of inside corners in y
ny = 9
# Extracting path of individual image stored in a given directory
resize_images("mobile images","resized",0.5)
images = glob.glob('mobile images/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    #ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    
    out = os.path.basename(fname).replace('.png', '_corners.png')
    cv2.imwrite(f'results_ex1/{out}', img)

cv2.destroyAllWindows()

h,w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

#Save parameters into numpy file
np.save("./camera_parameters/ret", ret)
np.save("./camera_parameters/mtx", mtx)
np.save("./camera_parameters/dist", dist)
np.save("./camera_parameters/rvecs", rvecs)
np.save("./camera_parameters/tvecs", tvecs)



#print(objpoints[0])
tot_error=0
total_points=0
for i in range(len(objpoints)):
    reprojected_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    reprojected_points=reprojected_points.reshape(-1,2)
    #print(reprojected_points)
    #print("i = ", i)
    #print(imgpoints[i])
    
i=0
for j in range(5) :
    error.append(np.sum(np.abs(imgpoints[i][j]-reprojected_points[j]), axis=0))
    #print(error)
print(error[0])
print(error[1])
print(error[2])
"""
print("\n--- Erreur de reprojection par image ---")
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    err = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    print(f"  Image {i:02d} : {err:.4f} px")
    mean_error += err
print(f"\nErreur moyenne de reprojection : {mean_error/len(objpoints):.4f} px")
print(f"RMS global (ret)              : {ret:.4f} px")

""" 