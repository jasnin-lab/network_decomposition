import argparse, time, sys

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from skimage import morphology
from scipy.stats import pearsonr
from src.pipeline.mask_processing import *
from src.pipeline.graph_constructor import *
from src.pipeline.synthetic_dataset_generation import *

GENERAL_PATH = "C:/Users/ant/PycharmProjects/bachelor_thesis/data/input_data/"
PLOT_PATH = "/data/plots/"

# ToDo: make parameters global variables
ITERATIONS = 5
UNIFORM_DILATION = True
FILES = ["B_simple", "CHI_simple", "J_BG_simple", "L_simple", "O_simple", "P_simple", "Q_simple", "R_simple",
         "S_simple", "T_simple", "X_simple", "Y_simple", "star_5", "star_9", "star_12", "star_12_thin",
         "star_12_normal", "star_12_thin_irregular", "star_irregular"]
file_type = ".jpg"

# Data type for each file
groups = {
    "letter": {"B_simple", "CHI_simple", "J_BG_simple", "L_simple", "O_simple", "P_simple", "Q_simple", "R_simple",
               "S_simple", "T_simple", "X_simple", "Y_simple"},
    "star": {"star_5", "star_9", "star_12", "star_12_thin", "star_12_normal_thickness", "star_12_thin_irregular",
             "star_irregular"}}


if __name__ == "__main__":
    # Instantiate storage data structures
    file_2_runtimes = {}  # runtime for each section -> generation + pipeline
    file_2_nv = {}  # N and V values for runtime analysis
    file_2_properties = {}  # Properties for the table in section "Results"

    for file in FILES:  # For each image/file in the provided directory path
        runtimes = []
        path_concat = [GENERAL_PATH, file, file_type]

        # Load image, binarize to a numpy array and define N for runtime analysis
        image = Image.open("".join(path_concat)).convert('L')
        binary_2d = (np.array(image) < 127).T
        K = np.count_nonzero(skeletonize(binary_2d, method='zhang'))
        V = (2 * 7 + 3) * binary_2d.shape[0] * binary_2d.shape[1]  # Z*Y*X, because uniform dilation is True
        file_2_nv.update({file: [K, V]})

        # Synthetic volume generation
        # Run 5 times and avg runtime (too little and highly variable between iterations)
        to_avg = []
        for i in range(ITERATIONS):
            start_time = time.perf_counter()
            mask = generate_3d_synthetic_volume(image_path="".join(path_concat), uniform_dilation=True, sphere_radius=7,
                                                mode=1, tilt=False, image_binarize_threshold=127)
            end_time = time.perf_counter()
            to_avg.append(round((end_time - start_time), 3))
        runtimes.append(sum(to_avg)/(1.0*ITERATIONS))

        # Count number of volume voxels
        VV = np.count_nonzero(mask)
        # print(f"And synthetic volume has shape {mask.shape}")

        # Closing
        start_time = time.perf_counter()
        mask = morphology.closing(mask, footprint=morphology.ball(radius=2))
        end_time = time.perf_counter()
        runtimes.append(round((end_time - start_time), 3))

        # Skeletonization
        start_time = time.perf_counter()
        mask_skeleton = skeletonization(mask)
        end_time = time.perf_counter()
        runtimes.append(round((end_time - start_time), 3))

        # Local Test
        start_time = time.perf_counter()
        intersections_nc, end_positions, final_intersections, intersection_map = local_test_nc(mask_skeleton)
        end_time = time.perf_counter()
        runtimes.append(round((end_time - start_time), 3))

        # Graph Construction
        start_time = time.perf_counter()
        graph_representation = graph_construction(mask_skeleton, np.argwhere(end_positions), final_intersections,
                                                  intersections_nc, mask, intersection_map, voxel_size=0)
        end_time = time.perf_counter()
        runtimes.append(round((end_time - start_time), 3))

        # Store runtimes and properties
        file_2_runtimes.update({file: runtimes})
        file_2_properties.update({file: [runtimes[0], (binary_2d.shape[0], binary_2d.shape[1]), K,
                                         (mask.shape[0], mask.shape[1], mask.shape[2]), VV]})
        # Runtime, Input image dims (X, Y); 2D skeletal voxels; Generated volume dims; Generated volume voxels

    # After acquiring the data - plot it
    # Sort in a specific way or the scatter plot to find correlations - or what N or V correlates stronger with runtime
    # Run one scatter plot for synthetic volume generation runtime vs N and another for vs V
    runtime = np.array([file_2_runtimes[file][0] for file in file_2_runtimes])
    V = np.array([file_2_nv[file][1] for file in file_2_runtimes])
    K = np.array([file_2_nv[file][0] for file in file_2_runtimes])

    # Runtime vs V
    r_V, _ = pearsonr(V, runtime)

    plt.figure()
    for group, files in groups.items():
        mask = np.array([file in files for file in file_2_runtimes])
        plt.scatter(V[mask] / 1e6, runtime[mask], label=group)
    plt.xlabel("V (number of volume voxels in millions)")
    plt.ylabel("Runtime in seconds")
    plt.title(f"Runtime vs V (Pearson r = {r_V:.3f})")
    plt.ticklabel_format(style="plain", axis="both")
    plt.legend()
    #plt.savefig("".join([PLOT_PATH, "/scatter_runtime_V.png"]))

    # Runtime vs K
    r_K, _ = pearsonr(K, runtime)

    plt.figure()
    for group, files in groups.items():
        mask = np.array([file in files for file in file_2_runtimes])
        plt.scatter(K[mask], runtime[mask], label=group)
    plt.xlabel("K (number of skeleton voxels)")
    plt.ylabel("Runtime in seconds")
    plt.title(f"Runtime vs K (Pearson r = {r_K:.3f})")
    plt.ticklabel_format(style="plain", axis="both")
    plt.legend()
    #plt.savefig("".join([PLOT_PATH, "/scatter_runtime_K.png"]))

    # Generate and store table
    table = pd.DataFrame.from_dict(
        file_2_properties,
        orient="index",
        columns=["Runtime in sec", "2D image (X, Y)", "Skeleton voxels", "3D Volume (X, Y, Z)", "Volume Voxels"])

    table = table.reset_index(names="File")

    # Experimental runtime of pipeline steps
    # Store for LaTeX
    #table.to_latex("C:/Users/ant/PycharmProjects/bachelor_thesis/data/generated_data/report/file_properties.tex", index=False)
    print(table.to_string())

    # Obtain data
    avg_runtimes = []
    for i in range(5):  # this dataset has an extra measurement we want to not include
        if i == 0:
            continue
        avg = sum(file_2_runtimes[file][i] for file in file_2_runtimes) / len(file_2_runtimes.keys())
        avg_runtimes.append(avg)

    print(file_2_runtimes)
    print("-"*100)
    print(avg_runtimes)

    categories = ['Closing', 'Skeletonization', 'Skeleton Voxel\nClassification', 'Graph Construction']
    plt.figure(figsize=(8, 5))
    plt.bar(categories, avg_runtimes, color='#4C72B0', edgecolor='black')
    # plt.xticks(rotation=45, ha='right')
    bars = plt.bar(categories, avg_runtimes, color='#4C72B0', edgecolor='black')
    plt.bar_label(bars, fmt='%.2f', padding=3)
    plt.ylim(top=max(avg_runtimes) * 1.10)
    plt.title('Experimental Runtime for Synthetic Dataset')
    plt.xlabel('Pipeline Step')
    plt.ylabel('Runtime in Seconds')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    #plt.show()
    plt.savefig('C:/Users/ant/PycharmProjects/bachelor_thesis/data/generated_data/report/synthetic_experimental_runtime_22.jpg')

