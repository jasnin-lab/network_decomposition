"""
Load, process all 19 podosome tomograms, store their results
Load, transfer and process Tom's template matching picks to tomo dimensions
Compare my with Tom's picks with F1 score...
Define a hyperparameter how much distance btw picks is considered a match
"""

import sys, time, os, re, pickle, random
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy.stats import norm
from skimage import morphology
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture
from scipy.spatial.transform import Rotation as R

from src.pipeline.mask_processing import *
from src.pipeline.graph_constructor import *
from src.pipeline.properties_computation.angles_at_intersections import *

# START_PATH = "C:/Users/ant/PycharmProjects/bachelor_thesis/data/"
START_PATH = "D:/BA_data/"

# GENERAL_INPUT_PATH = "D:/BA_data/Tom_data/segmentations/"
GENERAL_INPUT_PATH = "D:/BA_data/Tom_data/segmentations/"
GENERAL_MASK_PATH = f"{START_PATH}podosome_19_tomos/arp23/"


file_names = ["Position_1_Vol.nii", "Position_2_Vol.nii", "Position_8_Vol.nii", "Position_8_2_Vol.nii",
              "Position_12_2_Vol.nii", "Position_12_Vol.nii", "Position_30_Vol.nii", "Position_31_2_Vol.nii",
              "Position_32_2_Vol.nii", "Position_32_Vol.nii", "Position_37_2_Vol.nii", "Position_40_2_Vol.nii",
              "Position_40_Vol.nii", "Position_42_Vol.nii", "Position_44_Vol.nii", "Position_45_Vol.nii",
              "Position_66_Vol.nii", "Position_83_2_Vol.nii", "Position_86_Vol.nii"]


# file_names = ["Position_1_Vol.nii", "Position_2_Vol.nii", "Position_8_Vol.nii", "Position_8_2_Vol.nii",
#              "Position_12_2_Vol.nii", "Position_12_Vol.nii", "Position_30_Vol.nii", "Position_31_2_Vol.nii"]

# file_names = ["Position_1_Vol.nii"]

if __name__ == "__main__":
    runtimes = []
    # "file_name" : runtime for [closing, skeletonization, local test, graph construction]
    indexes = random.sample(range(0, len(file_names)), 5)  # [3, 6, 10, 14, 7] - for data in thesis
    for i in indexes:
        file = file_names[i]
        runtime = []

        original_nii = nib.load(f"{GENERAL_INPUT_PATH}{file}")
        mask = original_nii.get_fdata()

        print("Morphological Closing Operation:")
        start_time = time.perf_counter()
        mask = morphology.closing(mask, footprint=morphology.ball(radius=2))
        end_time = time.perf_counter()
        print(f"Morphological Closing done in {(end_time - start_time):.2f}s")
        runtime.append(end_time - start_time)

        print("Skeletonization:")
        start_time = time.perf_counter()
        mask_skeleton = skeletonization(mask)
        end_time = time.perf_counter()
        print(f"Skeletonization done in {(end_time - start_time):.2f}s")
        runtime.append(end_time - start_time)

        print("Local Test (Find Intersections and End Points):")
        start_time = time.perf_counter()
        intersections_nc, end_positions, final_intersections, intersection_map = local_test_nc(mask_skeleton)
        end_time = time.perf_counter()
        print(f"Local Test done in {(end_time - start_time):.2f}s")
        runtime.append(end_time - start_time)

        print("Graph Construction:")
        start_time = time.perf_counter()
        graph_representation = graph_construction(mask_skeleton, np.argwhere(end_positions), final_intersections,
                                                  intersections_nc, mask, intersection_map, None)
        end_time = time.perf_counter()
        print(f"Graph Construction done in {(end_time - start_time):.2f}s")
        runtime.append(end_time - start_time)

        runtimes.append(runtime)

    # Average runtime values
    categories = ['Closing', 'Skeletonization', 'Skeleton Voxel\nClassification', 'Graph Construction']

    avg_runtime = []
    for i in range(len(categories)):  # 4 is the number of categories
        avg = sum(measurement[i] for measurement in runtimes)/len(runtimes)
        avg_runtime.append(avg)

    # Bar graph
    plt.figure(figsize=(8, 5))
    plt.bar(categories, avg_runtime, color='#4C72B0', edgecolor='black')
    # plt.xticks(rotation=45, ha='right')
    bars = plt.bar(categories, avg_runtime, color='#4C72B0', edgecolor='black')
    plt.bar_label(bars, fmt='%.2f', padding=3)
    plt.ylim(top=max(avg_runtime) * 1.10)
    plt.title('Experimental Runtime for Cryo-ET')
    plt.xlabel('Pipeline Step')
    plt.ylabel('Runtime in Seconds')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.show()
    plt.savefig('C:/Users/ant/PycharmProjects/bachelor_thesis/data/generated_data/report/cryo_et_experimental_runtime_22.jpg')
