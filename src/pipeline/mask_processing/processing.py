"""
This script processes the input (semantic) segmentation mask and prepares the needed data for
next step of graph construction
This includes:
    - Volume skeletonization
    - Local test
        * Localizing potential intersections
        * Filtering for final intersections
        * Mapping FP to TP intersections
        * Localizing start points
Additionally, the rejected approach "Area-based" is also present
"""

import nrrd, mrcfile, random, sys
import numpy as np
from scipy.spatial import KDTree
from skimage.morphology import skeletonize
from sklearn.cluster import MeanShift
from sklearn.metrics import pairwise_distances_argmin
from src.pipeline.utils_vis import *


def skeletonization(volume: np.ndarray) -> np.ndarray:
    skeleton = skeletonize(volume, method='lee')
    return skeleton


def local_test_nc(skeleton: np.ndarray):
    # Create a new ndarray with same dimensions to store intersections
    skel_shape = skeleton.shape

    X_max, Y_max, Z_max = skel_shape
    # Create an NumPy array with identical dimensions to store the results from the local test
    # Data type uint8 is most suitable here since we only store 3 values:
    # 0 (background), 1 (in line or start position), or 2 (potential intersection).
    local_test_result = np.zeros(skel_shape, dtype=np.uint8)
    end_positions = np.zeros(skel_shape, dtype=np.uint8)

    # gather all coordinates with 1 = all voxels part of the skeleton
    skeleton_coords = np.where(skeleton)

    # zip(*...) groups the corresponding z, y, x coordinates together
    for x, y, z in zip(*skeleton_coords):
        # Define 3x3x3 boundaries with safe edge clipping so we don't look after the array edges
        x_min = max(0, x - 1)
        x_end = min(X_max, x + 2)

        y_min = max(0, y - 1)
        y_end = min(Y_max, y + 2)

        z_min = max(0, z - 1)
        z_end = min(Z_max, z + 2)

        # Extract the local 3x3x3 grid
        local_grid = skeleton[x_min:x_end, y_min:y_end, z_min:z_end]

        # Count direct skeleton neighbors (subtract the voxel in the middle)
        neighbor_count = np.sum(local_grid) - 1

        # Decide value based on the number of neighbors
        if neighbor_count == 1:
            local_test_result[x, y, z] = 1
            # Save as start/end position
            end_positions[x, y, z] = 1
        elif neighbor_count == 2:
            local_test_result[x, y, z] = 1
        elif neighbor_count >= 3:
            local_test_result[x, y, z] = 2

    # Filter intersections to contain only final ones
    potential_intersections = np.where(local_test_result == 2, 1, 0)
    intersection_coords = np.argwhere(potential_intersections)

    intersection_map = {}
    if len(intersection_coords) == 0:  # O and other have no intersections
        filtered_intersections = []
    else:
        # Apply MeanShift as postprocessing
        ms = MeanShift(bandwidth=2.0)
        ms.fit(intersection_coords)

        # Find sample points closest to centroids
        closest_indices = pairwise_distances_argmin(ms.cluster_centers_, intersection_coords)
        filtered_intersections = intersection_coords[closest_indices]

        # Use MeanShift labels to assign each FP intersection its TP intersection
        for point, label in zip(intersection_coords, ms.labels_):
            intersection_map[tuple(point)] = tuple(filtered_intersections[label])

    return local_test_result, end_positions, filtered_intersections, intersection_map


def test_triangle_area(skeleton_3d: np.ndarray, radius=3.0, area_threshold=2, samples=5):
    """
    Finds intersections by picking 3 random points within a given radius
    and checking whether the area of the resulting triangle is below or above a threshold
    """
    # Initialize results array (0 = background, 1 = line, 2 = intersection)
    result_map = np.zeros_like(skeleton_3d, dtype=np.uint8)

    # Get coordinates of all True voxels in the skeleton
    coords = np.argwhere(skeleton_3d)

    if len(coords) == 0:
        return result_map

    # # scipy.spatial.KDTree - a tool designed to give all the other points within R distance from a single point
    tree = KDTree(coords)

    for center_voxel in coords:
        # Find all skeleton voxel indices within the specified Euclidean radius
        neighbor_indices = tree.query_ball_point(center_voxel, r=radius)

        # We need at least 3 points to form a triangle
        if len(neighbor_indices) < 3:
            # Not enough neighbors, likely a start/end point
            result_map[tuple(center_voxel)] = 1
            continue

        local_points = coords[neighbor_indices]
        is_intersection = False

        # Repeat test to ensure intersections are found
        for _ in range(samples):
            # Pick 3 unique random points from the local neighborhood
            p1, p2, p3 = random.sample(list(local_points), 3)

            area = calculate_triangle_area(p1, p2, p3)

            if area > area_threshold:
                is_intersection = True
                break  # No need to keep sampling

        if is_intersection:
            result_map[tuple(center_voxel)] = 2  # An intersection is predicted at the current position
        else:
            result_map[tuple(center_voxel)] = 1

    return result_map


def calculate_triangle_area(p1, p2, p3):
    """ Calculates the 3D area of the triangle formed by three points. """
    v1 = p2 - p1
    v2 = p3 - p1

    # Area is half the magnitude of the cross product of the two vectors
    return 0.5 * np.linalg.norm(np.cross(v1, v2))
