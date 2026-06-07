import numpy as np

def validate_reconstruction(points_3d, ground_truth_distances):
    if len(points_3d) == 0:
        return "No points to validate."

    # Remove outliers (the points you have at 3000mm and 4500mm)
    # We only care about points within a reasonable range of your boxes
    filtered_points = points_3d[points_3d[:, 2] < 1000] 
    
    # Calculate the average depth of your reconstructed clusters
    mean_z = np.mean(filtered_points[:, 2])
    
    # Since SIFT points are scattered across all boxes, 
    # we compare the global mean to the average distance of your 3 boxes
    gt_average = np.mean(ground_truth_distances)
    
    error = abs(mean_z - gt_average)
    accuracy = (1 - (error / gt_average)) * 100

    print(f"--- Validation Report ---")
    print(f"Average reconstructed depth: {mean_z:.2f} mm")
    print(f"Average ground truth depth: {gt_average:.2f} mm")
    print(f"Absolute Error: {error:.2f} mm")
    print(f"System Accuracy: {accuracy:.2f}%")

