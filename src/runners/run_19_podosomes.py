"""
Load, process all 19 podosome tomograms, store their results
Load, transfer and process Tom's template matching picks to tomo dimensions
Compare my with Tom's picks with F1 score...
Define a hyperparameter how much distance btw picks is considered a match
"""

import sys, time, os, re, pickle
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy.stats import norm
from skimage import morphology
from collections import Counter
import matplotlib.pyplot as plt
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
GENERAL_SKELETON_PATH = f"{START_PATH}podosome_19_tomos/skeletons/"
GENERAL_MASK_PATH = f"{START_PATH}podosome_19_tomos/arp23/"
GENERAL_PLOT_PATH = f"{START_PATH}podosome_19_tomos/plots/"
GENERAL_VALIDATION_MASK_PATH = f"{START_PATH}podosome_19_tomos/structure_masks_15/"  # TP_FP_masks or structure_masks_20
STAR_FILEPATH = f"{START_PATH}podosome_19_tomos/arp23_predictions_empty.star"  # change depending on test run
GRAPHS_PATH = "D:/BA_data/podosome_19_tomos/graphs/"
STAR = True

MATCH_THRESHOLD = 20
INITIAL_GMM_MEANS = np.array([[8.0], [100.0]])
PERCENTILE = 0.90
FILAMENT_LENGTH_THRESHOLD = 10
# I believe this parameter should be a hyperparameter and be based on biological knowledge, if there is such...
# idk if there is a general value; is probably case-specific  - only if reference is provided - not usually the case

file_names = ["Position_1_Vol.nii", "Position_2_Vol.nii", "Position_8_Vol.nii", "Position_8_2_Vol.nii",
              "Position_12_2_Vol.nii", "Position_12_Vol.nii", "Position_30_Vol.nii", "Position_31_2_Vol.nii",
              "Position_32_2_Vol.nii", "Position_32_Vol.nii", "Position_37_2_Vol.nii", "Position_40_2_Vol.nii",
              "Position_40_Vol.nii", "Position_42_Vol.nii", "Position_44_Vol.nii", "Position_45_Vol.nii",
              "Position_66_Vol.nii", "Position_83_2_Vol.nii", "Position_86_Vol.nii"]

#file_names = ["Position_1_Vol.nii", "Position_2_Vol.nii", "Position_8_Vol.nii", "Position_8_2_Vol.nii",
#              "Position_12_2_Vol.nii", "Position_12_Vol.nii", "Position_30_Vol.nii", "Position_31_2_Vol.nii"]

#file_names = ["Position_1_Vol.nii"]


def plot_distribution_distances(list_distances, title, output_path, data_threshold=0.0, x_axis_label: str=None):
    if not len(list_distances):
        print(f"No distances to plot for {title}")
        return

    plt.figure(figsize=(14, 6))

    min_dist = int(np.floor(min(list_distances)))
    max_dist = int(np.ceil(max(list_distances)))

    # --- 1. Define custom bins ---

    if max_dist > 90:
        fine_bins = np.arange(min_dist, 91, 1)
        coarse_bins = np.linspace(90, max_dist, 8).astype(int)
        custom_bins = np.concatenate((fine_bins, coarse_bins[1:]))
    else:
        custom_bins = np.arange(min_dist, max_dist + 1, 1)

    # --- 2. Calculate histogram manually (do not plot yet) ---
    # np.histogram gives us the counts and the exact bin edges,
    # but does not force a linear plot!
    counts, bin_edges = np.histogram(list_distances, bins=custom_bins)

    # --- 3. Create equal-width positions for the bars ---
    # This simply creates an array like [0, 1, 2, ... 59]
    x_positions = np.arange(len(counts))

    # --- 4. Plot using a categorical Bar Chart ---
    # Every bar is now exactly width=1.0 visually
    bars = plt.bar(x_positions, counts, width=1.0, color='royalblue', edgecolor='black', alpha=0.8)

    # --- 5. Format X-Axis Labels ---

    ticks_to_show = []
    labels_to_show = []

    for i, edge in enumerate(bin_edges[:-1]):
        if edge < 40:
            ticks_to_show.append(i)
            labels_to_show.append(str(edge))
        elif edge < 90:
            # Show every 5th label for the detailed section
            # if edge == min_dist or edge % 3 == 0:
            ticks_to_show.append(i)
            labels_to_show.append(str(edge))
        else:
            # For the grouped section, label it as a range so it's obvious!
            next_edge = bin_edges[i+1]
            ticks_to_show.append(i)
            labels_to_show.append(f"{edge}-{next_edge}")

    plt.xticks(ticks_to_show, labels_to_show, rotation=45, fontsize=9)
    plt.xlim(-0.5, len(counts) - 0.5)  # Snap perfectly to the first and last bar
    """

    # --- 5. Format X-Axis Labels ---
    ticks_to_show = []
    labels_to_show = []

    for i, edge in enumerate(bin_edges[:-1]):
        ticks_to_show.append(i)
        # We just convert the bin edge directly to an integer and string
        labels_to_show.append(str(int(edge)))

    plt.xticks(ticks_to_show, labels_to_show, rotation=45, fontsize=9)
    plt.xlim(-0.5, len(counts) - 0.5)  # Snap perfectly to the first and last bar
    """

    # --- 6. Add the exact count above each bar ---
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)),
                     ha='center', va='bottom', fontsize=9, color='black')

    # Add labels, title, and grid
    plt.title(title, fontsize=14)
    plt.xlabel(x_axis_label, fontsize=12)  # "Distance (in voxels)"
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    if PERCENTILE != 0.0 and data_threshold != 0.0:
        plt.axvline(data_threshold - min_dist - 0.5, color='red', linestyle='--',
                    label=f'{int(PERCENTILE * 100)}th Percentile Cutoff ({data_threshold:.1f})')
        plt.legend()
    plt.savefig(output_path)
    plt.close()


def plot_gmm_distribution(list_distances, gmm, data_threshold, title, output_path):
    """
    Plots a linear histogram overlaid with the fitted Gaussian Mixture Model curves.
    """
    data = np.array(list_distances)
    if len(data) == 0:
        print(f"No distances to plot for {title}")
        return

    plt.figure(figsize=(10, 6))

    # --- 1. Plot the linear histogram ---
    # MUST use density=True so the histogram area sums to 1 (matching the PDF curves)
    plt.hist(data, bins=60, density=True, alpha=0.5, color='gray',
             edgecolor='black', label='Raw Data')

    # --- 2. Create a continuous X-axis for smooth curves ---
    x_min, x_max = data.min(), data.max()
    x_space = np.linspace(x_min, x_max, 1000).reshape(-1, 1)

    # --- 3. Plot the overall GMM combined curve ---
    # score_samples returns the log probability, so we use np.exp to get the true probability
    logprob = gmm.score_samples(x_space)
    pdf = np.exp(logprob)
    plt.plot(x_space, pdf, color='black', linewidth=2, label='Combined GMM Fit')

    # --- 4. Plot the individual Gaussian components ---
    colors = ['royalblue', 'darkorange']
    for i in range(gmm.n_components):
        mean = gmm.means_[i, 0]
        std = np.sqrt(gmm.covariances_[i, 0, 0])
        weight = gmm.weights_[i]

        # An individual GMM component is a standard normal distribution scaled by its weight
        component_pdf = weight * norm.pdf(x_space, mean, std)

        # Flatten x_space for matplotlib plotting
        plt.plot(x_space.flatten(), component_pdf.flatten(), linestyle='--',
                 color=colors[i % len(colors)], linewidth=2,
                 label=f'Component {i + 1} (Mean={mean:.1f})')

    # --- 5. Plot your calculated threshold ---
    if data_threshold > 0:
        plt.axvline(data_threshold, color='red', linestyle='-', linewidth=2,
                    label=f'{int(PERCENTILE * 100)}th Percentile Cutoff ({data_threshold:.1f})')

    # --- 6. Formatting ---
    plt.title(title, fontsize=14)
    plt.xlabel("Distance (in voxels)", fontsize=12)
    plt.ylabel("Density (Probability)", fontsize=12)

    # Place legend outside the plot if it covers the data
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Helper function to safely get angles regardless of tuple order
def get_angle(fi1, fi2, angle_dict):
    if (fi1, fi2) in angle_dict:
        return angle_dict[(fi1, fi2)]
    elif (fi2, fi1) in angle_dict:
        return angle_dict[(fi2, fi1)]
    else:
        raise ValueError(f"Angle between {fi1} and {fi2} not found in dictionary.")


def append_2_star_file_4_sta(graph, inters_candidates, volume_name, vol_shape_dict):
    """
    Whole idea is to process the picks and append the information/data for them at the end of the start file
    """
    # Using 'a' opens the file for appending at the end
    with open(STAR_FILEPATH, 'a') as f:
        for node_id in inters_candidates:
            # get the node
            current_node = graph.nodes[node_id]

            # get filaments and angles and extract mother and daughter
            pairs = list(current_node["angles"].items())

            pair_70 = min(pairs, key=lambda x: abs(x[1] - 70))
            fil_1, fil_2 = pair_70[0]
            print(f"For intersection {current_node['coords']} we get the filaments at 70 angle {fil_1} and {fil_2}")

            if len(current_node["connections"].keys()) != 3:
                raise Exception("Should not happen - a potential Arp23 with not exactly 3 connecting filaments")

            fil_3 = (current_node["connections"].keys() - set([fil_1, fil_2])).pop()

            angle_f1_f3 = get_angle(fil_1, fil_3, current_node["angles"])
            angle_f2_f3 = get_angle(fil_2, fil_3, current_node["angles"])

            if angle_f1_f3 < angle_f2_f3:
                mother = fil_2
                daughter = fil_1
            else:
                mother = fil_1
                daughter = fil_2

            mother_unit_vector = current_node["connections"][mother]
            daughter_unit_vector = current_node["connections"][daughter]
            print(f"Mother filament unit vector of direction: ", mother_unit_vector)
            print(f"Daughter filament unit vector of direction: ", daughter_unit_vector)

            if np.linalg.norm(mother_unit_vector) == 0.0 or np.linalg.norm(daughter_unit_vector) == 0.0:
                print(f"Skipping intersection {current_node['coords']} due to zero-length vector.")
                continue

            #print(f"For intersection {current_node['coords']} we get mother filament {mother} and daughter {daughter}")
            #print(f"The mother filament has path: {graph.graph['filament_paths'][mother].data[:10]}")
            #print(f"The daughter filament has path: {graph.graph['filament_paths'][daughter].data[:10]}")
            #print(f"The other part of mother filament has path: {graph.graph['filament_paths'][fil_3].data[:10]}")

            line = generate_star_line(intersection_xyz=current_node['coords'],
                                      mother_vec_xyz=mother_unit_vector,
                                      daughter_vec_xyz=daughter_unit_vector,
                                      volume_name=volume_name,
                                      volume_shape_dictionary=vol_shape_dict)
            f.write(line + "\n")


def collect_angles(graph, intersections_test) -> []:
    """ Collect individual """
    all_angles = []
    all_fil_lens = []
    min_fil_lens = []
    all_daughter_lens = []
    # Using 'a' opens the file for appending at the end
    for node_id in intersections_test:  # list of node ids
        # get the node
        current_node = graph.nodes[node_id]

        # get filaments and angles and extract mother and daughter
        pairs = list(current_node["angles"].items())

        pair_70 = min(pairs, key=lambda x: abs(x[1] - 70))  # get filaments with angle closest to 70
        # the same way mother and daughter filaments are chosen
        fil_1, fil_2 = pair_70[0]
        angle_of_interest = get_angle(fil_1, fil_2, current_node['angles'])
        print("Angle of interest ", angle_of_interest)

        # Determine daughter and get it's length
        fil_3 = (current_node["connections"].keys() - set([fil_1, fil_2])).pop()

        angle_f1_f3 = get_angle(fil_1, fil_3, current_node["angles"])
        angle_f2_f3 = get_angle(fil_2, fil_3, current_node["angles"])

        if angle_f1_f3 < angle_f2_f3:
            daughter = fil_1
        else:
            daughter = fil_2
        all_daughter_lens.append(graph.graph["filament_paths"][daughter].length())

        all_angles.append(angle_of_interest)

        # get filament lengths
        # all_fil = current_node['connections'].keys()
        all_fil_lens.extend([graph.graph["filament_paths"][fill].length() for fill in current_node['connections'].keys()])
        min_fil_lens.append(min([graph.graph["filament_paths"][fill].length() for fill in current_node['connections'].keys()]))

    return all_angles, all_fil_lens, min_fil_lens, all_daughter_lens


def generate_star_line(intersection_xyz, mother_vec_xyz, daughter_vec_xyz, volume_name, volume_shape_dictionary):
    """
    Converts (Z, Y, X) inputs into (X, Y, Z) Euler angles and returns a formatted STAR file line.
    """

    # 1. Extract coordinates in (X, Y, Z) order
    # Also convert them to Angstrom coordinates suitable for STA
    # x = float(intersection_zyx[2]*8.24 - volume_shape_dictionary[volume_name][0])
    # y = float(intersection_zyx[1]*8.24 - volume_shape_dictionary[volume_name][1])
    # z = float(intersection_zyx[0]*8.24 - volume_shape_dictionary[volume_name][2])
    x = float(intersection_xyz[0])
    y = float(intersection_xyz[1])
    z = float(intersection_xyz[2])

    # 2. Convert unit vectors from (Z, Y, X) to (X, Y, Z)
    m_xyz = np.array([mother_vec_xyz[0], mother_vec_xyz[1], mother_vec_xyz[2]], dtype=float)
    d_xyz = np.array([daughter_vec_xyz[0], daughter_vec_xyz[1], daughter_vec_xyz[2]], dtype=float)

    # 3. Calculate local orthonormal axes
    z_axis = m_xyz / np.linalg.norm(m_xyz)

    y_axis = np.cross(z_axis, d_xyz)
    y_axis = y_axis / np.linalg.norm(y_axis)

    x_axis = np.cross(y_axis, z_axis)

    # 4. Generate Rotation Matrix and extract Z-Y-Z Euler angles
    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
    rotation = R.from_matrix(rot_matrix)
    rot, tilt, psi = rotation.as_euler('ZYZ', degrees=True)  # PyTom, for Dennis in ZYZ

    # 5. Format the line for the STAR file (X, Y, Z, Rot, Tilt, Psi, TomoName)
    # Using 12 characters and 6 (5 now) decimal places is standard for RELION star files
    #star_line = f"{x:12.6f} {float((intersection_xyz[0] - volume_shape_dictionary[volume_name][0] / 2) * 8.24):12.6f} " \
    #            f"{y:12.6f} {float((intersection_xyz[1] - volume_shape_dictionary[volume_name][1] / 2) * 8.24):12.6f} " \
    #            f"{z:12.6f} {float((intersection_xyz[2] - volume_shape_dictionary[volume_name][2] / 2) * 8.24):12.6f} " \
    #            f"{rot:12.6f} {tilt:12.6f} {psi:12.6f}   {volume_name}"

    star_line = f"{float((intersection_xyz[0] - volume_shape_dictionary[volume_name][0] / 2) * 8.24):12.6f} " \
                f"{float((intersection_xyz[1] - volume_shape_dictionary[volume_name][1] / 2) * 8.24):12.6f} " \
                f"{float((intersection_xyz[2] - volume_shape_dictionary[volume_name][2] / 2) * 8.24):12.6f} " \
                f"{rot:12.6f} {tilt:12.6f} {psi:12.6f}   {volume_name}"

    return star_line


def store_skeleton_nii(skeleton, original_nii_path, output_nii_path):
    skeleton_to_store = skeleton.astype(np.uint8)

    original = nib.load(original_nii_path)

    new_image = nib.Nifti1Image(skeleton_to_store, original.affine, header=original.header.copy())
    nib.save(new_image, output_nii_path)
    print(f"Successfully saved perfectly aligned NRRD mask to {output_nii_path}")


def initialize_star_file():
    """Creates a fresh STAR file and writes the header."""
    # Using 'w' overwrites any existing file with this name
    with open(STAR_FILEPATH, 'w') as f:
        f.write("data_\n\n")
        f.write("loop_\n")
        f.write("_rlnCoordinateX #1\n")
        f.write("_rlnCoordinateY #2\n")
        f.write("_rlnCoordinateZ #3\n")
        f.write("_rlnAngleRot #4\n")
        f.write("_rlnAngleTilt #5\n")
        f.write("_rlnAnglePsi #6\n")
        f.write("_rlnTomoName #7\n")
    print(f"Initialized new STAR file")


def filter_intersections(graph: nx.Graph, picks: [], len_threshold: int):
    filtered_intersections = []
    filtered_nodes = []
    for node_id, attributes in graph.nodes(data=True):
        if attributes.get("coords") in picks:  # filter picked intersections
            # safely get connections, defaulting to empty list if key is missing
            connections = attributes.get("connections", [])

            # Check if ALL connected filaments are >= threshold  (instead of a classic "boolean flag" pattern)
            if all(graph.graph["filament_paths"][filam].length() >= len_threshold for filam in connections):
                filtered_intersections.append(attributes.get("coords"))
                filtered_nodes.append(node_id)

    return filtered_intersections, filtered_nodes


if __name__ == "__main__":
    # Enable direct check of volume size
    file_2_size = {}
    for file_name in file_names:
        shape = nib.load(f"{GENERAL_INPUT_PATH}{file_name}").header.get_data_shape()
        file_2_size.update({file_name: shape})

    # Preprocess Tom's results to be able to use them directly to compute the comparison
    df_particles = pd.read_excel("C:/Users/ant/PycharmProjects/bachelor_thesis/data/podosome_19_tomos/particles_17_tomos.xlsx",
                                 sheet_name="Sheet1", engine="openpyxl")

    # --- Assign each particle to a layer using the segmentation volumes ---
    tomo_2_tm_picks = {}
    for tomo_name, group in df_particles.groupby("tomogram_name"):
        tomo = re.search(r'([^/]+)\.[^./]+$', f"{tomo_name}").group(1)

        if f"{tomo}_Vol.nii" not in file_2_size.keys():
            continue

        volume_shape = file_2_size[f"{tomo}_Vol.nii"]

        group = group.copy()

        group["CoordinateX"] = group["rlnCoordinateX"] / (4 * 2.06) + volume_shape[0] / 2
        group["CoordinateY"] = group["rlnCoordinateY"] / (4 * 2.06) + volume_shape[1] / 2
        group["CoordinateZ"] = group["rlnCoordinateZ"] / (4 * 2.06) + volume_shape[2] / 2

        coords_x = group["CoordinateX"].round().astype(int)
        coords_y = group["CoordinateY"].round().astype(int)
        coords_z = group["CoordinateZ"].round().astype(int)

        z_max = volume_shape[2] - 1
        y_max = volume_shape[1] - 1
        x_max = volume_shape[0] - 1

        coords_x_clipped = coords_x.clip(0, x_max)
        coords_y_clipped = coords_y.clip(0, y_max)
        coords_z_clipped = coords_z.clip(0, z_max)

        group["in_volume"] = (
                (coords_x >= 0) & (coords_x <= x_max) &
                (coords_y >= 0) & (coords_y <= y_max) &
                (coords_z >= 0) & (coords_z <= z_max))

        # Keep only picks inside a volume
        valid = group[group["in_volume"]]

        # Convert coordinates to (x, y, z) tuples and store
        coordinates = list(zip(coords_x[valid.index], coords_y[valid.index], coords_z[valid.index]))  # stored as (X, Y, Z)

        tomo_2_tm_picks[f"{tomo}_Vol.nii"] = coordinates
        print(f"Tomo {tomo} has {len(coordinates)} template matching picks")

        # Make a mask and visualize to check results
        # mask_picks_nii(None, coordinates, f"{GENERAL_INPUT_PATH}{tomo}_Vol.nii", f"{GENERAL_MASK_PATH}{tomo}_tm.nii")

    print("Ok with tomo 2 tm_picks")

    runtimes = {}
    # "file_name" : runtime for [closing, skeletonization, local test, graph construction , angles computation]
    f1_scores = {}
    raw_scores = {}
    # initialize_star_file()  # and for each file, add results  # ToDo: change
    all_fp_fil_len = []
    all_angles_data = []
    all_filtered_fil_lengths = []
    min_fil_lengths = []
    all_dau_lens = []
    for file in file_names:
        runtime = []
        # Load nii file

        # If the graph file has already been saved; if so --> skip generating the graph and only load it
        if not Path(f"{GRAPHS_PATH}graph_{file.replace('_Vol.nii', '')}.pkl").is_file():
            original_nii = nib.load(f"{GENERAL_INPUT_PATH}{file}")
            mask = original_nii.get_fdata()
        
            # Load voxel_size
            voxel_size = original_nii.header.get_zooms()

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

            # Store skeleton for result evaluation
            #store_skeleton_nii(mask_skeleton, f"{GENERAL_INPUT_PATH}{file}",
            #               f"{GENERAL_SKELETON_PATH}{file.replace('_Vol.nii', '')}_skeleton.nii")

            print("Local Test (Find Intersections and End Points):")
            start_time = time.perf_counter()
            intersections_nc, end_positions, final_intersections, intersection_map = local_test_nc(mask_skeleton)
            end_time = time.perf_counter()
            print(f"Local Test done in {(end_time - start_time):.2f}s")
            runtime.append(end_time - start_time)

            print("Graph Construction:")
            start_time = time.perf_counter()
            graph_representation = graph_construction(mask_skeleton, np.argwhere(end_positions), final_intersections,
                                                      intersections_nc, mask, intersection_map, voxel_size)
            end_time = time.perf_counter()
            print(f"Graph Construction done in {(end_time - start_time):.2f}s")
            runtime.append(end_time - start_time)

            with open(f"{GRAPHS_PATH}graph_{file.replace('_Vol.nii', '')}.pkl", "wb") as f:
                pickle.dump(graph_representation, f)

        # Load graph representation to save time
        with open(f"{GRAPHS_PATH}graph_{file.replace('_Vol.nii', '')}.pkl", "rb") as f:
            graph_representation = pickle.load(f)

        print("Angles at Intersections:")
        start_time = time.perf_counter()
        compute_dirs_and_angles(graph=graph_representation, lower_limit=5, upper_limit=35)
        end_time = time.perf_counter()
        print(f"Angles at Intersections done in {(end_time - start_time):.2f}s")
        runtime.append(end_time - start_time)

        print("Filter Intersection Candidates:")
        start_time = time.perf_counter()
        # Filter candidates based on angle criteria
        arp23_candidates = get_intersection_candidates(graph_representation, (70, 110, 180), 20)
        # Get all intersections
        # arp23_candidates = [n for n, attr in graph_representation.nodes(data=True)
        #                     if attr.get('node_type') == "intersection"]
        end_time = time.perf_counter()
        print(f"Filter Intersection Candidates done in {(end_time - start_time):.2f}s")
        runtime.append(end_time - start_time)

        # Store runtimes
        runtimes.update({file: runtime})

        # Create a mask out of the predicted Arp23s
        # ToDo: only this is - nope, this is unfiltered data...
        #mask_picks_nii(graph_representation, arp23_candidates, f"{GENERAL_INPUT_PATH}{file}",
        #               f"{GENERAL_MASK_PATH}{file.replace('_Vol.nii', '')}_prediction.nii")


        # Iterate intersections, compute Euler angles and append predictions in STAR for STA
        # Remove, stores all intersections, while we want the filtered ones
        # append_2_star_file_4_sta(graph_representation, arp23_candidates, file, file_2_size)  # ToDo: remove when computing the data for Dennis

        # The "arp23_candidates" are node id's of filtered intersection voxels
        """  --> Comparison with Tom's template matching picks
        For every point in set A (predicted picks), find the nearest point in B (Tom's picks). 
        If the Euclidean distance is ≤ some threshold, consider it a match.
        """
        tm_picks = tomo_2_tm_picks.get(file, []).copy()

        # General number comparison
        print(f"For tomo {file} we predict {len(arp23_candidates)} Arp23 picks, "
              f"while with tm we get {len(tm_picks)}")

        # Only for generating all picks masks and the whole .star
        if STAR:  #  not tm_picks - an empty list [] is considered False in a boolean context
            # Special handling of files without tm picks

            all_fil_lengths = []
            all_node_coords = []
            for node, attr in graph_representation.nodes(data=True):
                if node in arp23_candidates:  # or tp_coords to consider all TPs
                    all_node_coords.append(attr.get('coords'))
                    print("New fil")
                    for fil in attr.get("connections"):
                        all_fil_lengths.append(graph_representation.graph["filament_paths"][fil].length())
                        if graph_representation.graph["filament_paths"][fil].length() < 10:
                            print(f"Short one with length {graph_representation.graph['filament_paths'][fil].length()}")
                        else:
                            print("Passes threshold ", graph_representation.graph['filament_paths'][fil].length())

            print("All filament lengths in te list are ", len(all_fil_lengths))

            filtered_inters, filtered_node_ids = filter_intersections(graph_representation, all_node_coords, FILAMENT_LENGTH_THRESHOLD)

            # Store and visualize results
            print(f"Updated size of FPs: {len(filtered_inters)}")

            to_add, filtered_fil_lens, min_fil, dau_lens = collect_angles(graph_representation, filtered_node_ids)
            all_filtered_fil_lengths.extend(filtered_fil_lens)
            print(f"num angles to extend by {len(to_add)} compared to {len(filtered_inters)}")
            all_angles_data.extend(to_add)
            min_fil_lengths.extend(min_fil)
            all_dau_lens.extend(dau_lens)

            # Append to star file
            #append_2_star_file_4_sta(graph_representation, filtered_node_ids, file, file_2_size)

            # Plot distribution to check if all good
            plot_distribution_distances(all_fil_lengths, "Distribution of filament lengths",
                                        f"{GENERAL_PLOT_PATH}tests_for_final/{file.replace('_Vol.nii', '')}_fil_dis_histogram.png")

            original_nii = f"{GENERAL_INPUT_PATH}{file}"
            #mask_picks_nii(graph_rep=None,
            #               candidates=filtered_inters,
            #               original_nii_path=original_nii,
            #               output_nii_path=f"{GENERAL_VALIDATION_MASK_PATH}{file.replace('_Vol.nii', '')}_filtered_picks_mask.nii",  # ToDo: change
            #               radius=10)
            continue

        # Detailed per pick comparison
        tree = cKDTree(tm_picks)  # use a tree to save time and avoid n^2 runtime

        # Transfer picks from node ids to coordinates
        predicted_picks = [graph_representation.nodes[node_id]['coords'] for node_id in arp23_candidates]

        distances, indices = tree.query(predicted_picks)
        # Consider distance_upper_bound to "Return only neighbors within this distance. "
        matches = distances <= MATCH_THRESHOLD

        #plot_distribution_distances(distances, "Distribution of Distance between Predicted and TM's Arp2/3 picks",
        #                            f"{GENERAL_PLOT_PATH}distance_distribution/{file.replace('_Vol.nii', '')}_pred_tm_dis_histogram.png")

        # An index does not mean that it should be within the threshold, which is why we should actually use the matches
        matched_indices = indices[matches]
        print(f"Indices with size {indices.size}: {indices}")
        print(f"Matched indices with size {matched_indices.size}")

        # 1. True Positives: The number of UNIQUE reference points successfully hit
        TP = np.unique(matched_indices).size

        # 2. False Positives: Any prediction that isn't a TP is an FP
        FP = len(predicted_picks) - TP

        # 3. False Negatives: Any reference point that isn't a TP is an FN
        FN = len(tm_picks) - TP

        # 4. F1 Score
        f1 = TP / (TP + (FP + FN) / 2) if TP > 0 else 0.0

        print(f"Results of analysis: TP = {TP}, FP = {FP}, FN = {FN}")
        print(f"File {file} has F1 score: {f1}")
        print(f"For correctness check, all predicted are {len(arp23_candidates)}")
        print(f"For correctness check, all reference are {len(tm_picks)}")
        f1_scores.update({file: f1})
        raw_scores.update({file: [TP, FP, FN]})
        sys.exit()  # I only need the

        # Only for visualization purposes:
        # 2. Prepare lists to hold the final separated coordinates
        tp_coords = []
        fp_coords = []

        # To enforce 1-to-1 matching, we track which reference points are already "taken"
        claimed_refs = set()

        # To keep track of
        matched_pairs = {}

        # 3. Create a list of all data so we can sort it by distance
        # Format: (prediction_index, reference_index, distance)
        # Zip and sort from shortest distance to largest distance
        match_data = [(pred_idx, ref_idx, dist) for pred_idx, (dist, ref_idx) in enumerate(zip(distances, indices))]
        match_data.sort(key=lambda x: x[2])
        # Sort from smallest distance to largest distance
        # This guarantees that the BEST predictions get first claim on the reference points

        # 4. Iterate and strictly assign TP vs FP
        true_tp_coords = []
        for pred_idx, ref_idx, dist in match_data:
            coord = predicted_picks[pred_idx]

            if dist <= MATCH_THRESHOLD:
                if ref_idx not in claimed_refs:
                    # It's within threshold AND the reference is free! -> True Positive
                    claimed_refs.add(ref_idx)
                    tp_coords.append(coord)

                    if dist < 10:  # threshold for true TPs
                        true_tp_coords.append(coord)

                    # Get the reference coordinate and save the pair
                    ref_coord = tm_picks[ref_idx]
                    # matched_pairs[tuple(coord)] = tuple(ref_coord)
                    matched_pairs[tuple(coord)] = (tuple(ref_coord), float(dist))
                else:
                    # Within threshold, but a closer prediction already stole this reference -> False Positive
                    fp_coords.append(coord)
            else:
                # Too far away -> False Positive
                fp_coords.append(coord)

        # I have the coordinates for the FP and TP --> search for their nodes based on that
        # Assign node ids to each
        tp_nodes = []
        tp_fil_lengths = []
        fp_nodes = []
        fp_fil_lengths = []
        tp_picks = 0
        fp_picks = 0
        for node, attr in graph_representation.nodes(data=True):
            if attr.get("coords") in true_tp_coords:  # or tp_coords to consider all TPs
                tp_nodes.append(node)  # Consider only TP with distance < 10
                tp_picks += 1
                for fil in attr.get("connections"):
                    tp_fil_lengths.append(graph_representation.graph["filament_paths"][fil].length())
                    print(f"Adding TP fil with length {graph_representation.graph['filament_paths'][fil].length()}")
                    if graph_representation.graph["filament_paths"][fil].length() < 20:
                        print(f"Adding a filament with length {graph_representation.graph['filament_paths'][fil].length()}"
                              f" that has direction {graph_representation.graph['filament_paths'][fil].dir()}")

            elif attr.get("coords") in fp_coords:
                fp_nodes.append(node)
                fp_picks += 1
                for fil in attr.get("connections"):
                    fp_fil_lengths.append(graph_representation.graph["filament_paths"][fil].length())
                    print(f"Adding FP fil with length {graph_representation.graph['filament_paths'][fil].length()}")
                    if graph_representation.graph["filament_paths"][fil].length() < 20:
                        print(f"Adding a filament with length {graph_representation.graph['filament_paths'][fil].length()}"
                              f" that has direction {graph_representation.graph['filament_paths'][fil].dir()}")

        print(f"Found {tp_picks} TP picks and {fp_picks} FP picks")
        # Plot filament length distributions
        #plot_distribution_distances(tp_fil_lengths, "Distribution of TP filament lengths",
        #                            f"{GENERAL_PLOT_PATH}filament_length_distribution/{file.replace('_Vol.nii', '')}_new_test_tp_fil_dis_histogram.png")
        #plot_distribution_distances(fp_fil_lengths, "Distribution of FP filament lengths",
        #                            f"{GENERAL_PLOT_PATH}filament_length_distribution/{file.replace('_Vol.nii', '')}_new_test_fp_fil_dis_histogram.png")

        # Apply Gaussian Mixture Models to define a (dynamic) threshold for filtering out artifact filaments
        # and consequently lower the FPs
        X = np.array(fp_fil_lengths).reshape(-1, 1)
        all_fp_fil_len.extend(fp_fil_lengths)

        # 2. Fit the Gaussian Mixture Model
        gmm = GaussianMixture(n_components=2, means_init=INITIAL_GMM_MEANS, random_state=42)
        gmm.fit(X)

        # 3. Identify the artifact component (the one with the smaller mean)
        # gmm.means_ returns an array of means for each component
        artifact_idx = np.argmin(gmm.means_[:, 0])

        # Extract the specific mean and standard deviation for the artifacts
        artifact_mean = gmm.means_[artifact_idx, 0]
        # Covariances are returned as variances (std squared), so we take the square root
        artifact_std = np.sqrt(gmm.covariances_[artifact_idx, 0, 0])

        # 4. Calculate the threshold (e.g., 90th percentile)
        # norm.ppf calculates the exact value at a given percentile for a normal distribution
        threshold = norm.ppf(PERCENTILE, loc=artifact_mean, scale=artifact_std)

        print(f"Artifact Distribution -> Mean: {artifact_mean:.2f}, Std Dev: {artifact_std:.2f}")
        print(f"Calculated {PERCENTILE * 100}th percentile threshold: {threshold:.2f}")

        # Visualize the cutoff
        """
        plt.hist(fp_fil_lengths, bins=100, density=True, alpha=0.6, color='gray', label='Raw Data')
        plt.axvline(threshold, color='red', linestyle='--',
                    label=f'{int(PERCENTILE * 100)}th Percentile Cutoff ({threshold:.1f})')
        plt.title("Filtering Artifacts via GMM Percentile")
        plt.xlabel("Filament Length")
        plt.ylabel("Density")
        plt.legend()
        plt.show()
        """
        # --------------------------
        #plot_distribution_distances(fp_fil_lengths, "Distribution of FP filament lengths",
        #                            f"{GENERAL_PLOT_PATH}filament_length_distribution/{file.replace('_Vol.nii', '')}_new_test_fp_fil_dis_histogram_GMM.png",
        #                            threshold)

        # Plot GMM
        #plot_gmm_distribution(fp_fil_lengths, gmm, threshold, "Gaussian Mixture Model of filaments of FP intersections",
        #                      f"{GENERAL_PLOT_PATH}filament_length_distribution/{file.replace('_Vol.nii', '')}_new_test_fp_fil_dis_histogram_GMM_distributions.png")

        filtered_fp_intersections, filtered_fp_nodes = filter_intersections(graph_representation, fp_nodes,
                                                                            FILAMENT_LENGTH_THRESHOLD)
        # ToDo: FILAMENT_LENGTH_THRESHOLD is a global variable; can be given without making it a parameter

        # Store and visualize results
        print(f"Updated size of FPs: {len(filtered_fp_intersections)}")
        print(f"Updated f1 score: {TP / (TP + (len(filtered_fp_intersections) + FN) / 2) if TP > 0 else 0.0}")
        original_nii = f"{GENERAL_INPUT_PATH}{file}"
        """
        mask_picks_nii(graph_rep=None,
                       candidates=filtered_fp_intersections,
                       original_nii_path=original_nii,
                       output_nii_path=f"{GENERAL_VALIDATION_MASK_PATH}{file.replace('_Vol.nii', '')}_filtered_FP_Mask_new_test.nii",  # ToDo: change
                       radius=10)
        """

        print(f"Total Predictions: {len(predicted_picks)}")
        print(f"Total TPs: {len(tp_coords)}")
        print(f"Total FPs: {len(fp_coords)}")

        # Extract false negatives
        fn_coords = []
        for ref_idx, ref_coord in enumerate(tm_picks):
            if ref_idx not in claimed_refs:
                fn_coords.append(ref_coord)

        # --- Print the matched pairs with distances ---
        """
        print("\n=== TP Matches (Predicted Coord -> Reference Coord | Distance) ===")
        for pred_c, (ref_c, dist) in matched_pairs.items():
            p_str = f"({pred_c[0]:.2f}, {pred_c[1]:.2f}, {pred_c[2]:.2f})"
            r_str = f"({ref_c[0]:.2f}, {ref_c[1]:.2f}, {ref_c[2]:.2f})"
            print(f"Predicted: {p_str}  -->  Reference: {r_str}  |  Distance: {dist:.2f}")
            print("-->")
            print()
        print("=================================================================\n")
        """

        # 5. Generate the three separate NIfTI masks
        original_nii = f"{GENERAL_INPUT_PATH}{file}"  # define your input path

        # ToDo: Store predictions AFTER filtering with GMM (filtered_fp_intersections + tp_coords...) check again logic
        # ToDo: provide those filtered ones for STA

        """
        # Generate the TP mask
        mask_picks_nii(graph_rep=None,
                       candidates=tp_coords,
                       original_nii_path=original_nii,
                       output_nii_path=f"{GENERAL_VALIDATION_MASK_PATH}{file.replace('_Vol.nii', '')}_TP_Mask.nii",
                       radius=10)
        """

        """
        # Generate the FP mask
        mask_picks_nii(graph_rep=None,
               candidates=fp_coords,
               original_nii_path=original_nii,
               output_nii_path=f"{GENERAL_VALIDATION_MASK_PATH}{file.replace('_Vol.nii', '')}_FP_Mask_new_test.nii",  # ToDo: change
               radius=10)

        mask_picks_niigraph_rep=None,
               candidates=fn_coords,
               original_nii_path=original_nii,
               output_nii_path=f"{GENERAL_VALIDATION_MASK_PATH}{file.replace('_Vol.nii', '')}_FN_Mask_new_test.nii",  # ToDo: change
               radius=10)
        """

    print("Done, print final statistics: ")
    print("Runtime measurements:")
    for file in runtimes.keys():
        print(f"For file {file} the runtimes are: {runtimes[file]}")
    print("-"*100)
    for file in f1_scores.keys():
        print(f"For file {file} the f1 scores are: {f1_scores[file]}")
        print(f"For file {file} the raw scores are: {raw_scores[file]}")
        print()
    print("-"*100)
    print("Average f1 score:")
    numeric_values = [v for v in f1_scores.values() if isinstance(v, (int, float))]
    if not len(numeric_values) == 0:
        print((sum(numeric_values) / len(numeric_values)))

    print("done")

    #plot_distribution_distances(all_angles_data, "Distribution of Angles Across Filtered Intersections",
    #                            f"{GENERAL_PLOT_PATH}angles_distribution_test_20_2.png", 0.0, "Angles (°)")

    #plot_distribution_distances(all_filtered_fil_lengths, "Distribution of filtered filament lengths",
    #                            f"{GENERAL_PLOT_PATH}/filtered_fil_lengths_test_20_2.png", 0.0, "Length")

    #plot_distribution_distances(min_fil_lengths, "Distribution of MINIMAL filtered filament lengths per filtered intersection",
    #                            f"{GENERAL_PLOT_PATH}/MIN_filtered_fil_lengths_20_2.png", 0.0, "Length")

    #plot_distribution_distances([k*8.24 for k in all_dau_lens],
    #                            "Distribution of Daughter Filament Lengths per Filtered Intersection",
    #                            f"{GENERAL_PLOT_PATH}daughter_filtered_fil_lengths_20_2.png", 0.0, "Length (Ångström)")
    plot_distribution_distances(all_dau_lens,
                                "Distribution of Daughter Filament Lengths per Filtered Intersection",
                                f"{GENERAL_PLOT_PATH}daughter_filtered_fil_voxel_lengths_20_new.png", 0.0, "Length (voxel)")


    sys.exit()
    # Examine overall fil length threshold:
    X = np.array(all_fp_fil_len).reshape(-1, 1)

    # 2. Fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=2, means_init=INITIAL_GMM_MEANS, random_state=42)
    gmm.fit(X)

    # 3. Identify the artifact component (the one with the smaller mean)
    # gmm.means_ returns an array of means for each component
    artifact_idx = np.argmin(gmm.means_[:, 0])

    # Extract the specific mean and standard deviation for the artifacts
    artifact_mean = gmm.means_[artifact_idx, 0]
    # Covariances are returned as variances (std squared), so we take the square root
    artifact_std = np.sqrt(gmm.covariances_[artifact_idx, 0, 0])

    # 4. Calculate the threshold (e.g., 90th percentile)
    # norm.ppf calculates the exact value at a given percentile for a normal distribution
    threshold = norm.ppf(PERCENTILE, loc=artifact_mean, scale=artifact_std)

    print(f"Artifact Distribution -> Mean: {artifact_mean:.2f}, Std Dev: {artifact_std:.2f}")
    print(f"Calculated {PERCENTILE * 100}th percentile threshold for ALL FP filament lengths: {threshold:.2f}")

    # Visualize the cutoff
    plt.hist(all_fp_fil_len, bins=100, density=True, alpha=0.6, color='gray', label='Raw Data')
    plt.axvline(threshold, color='red', linestyle='--',
                label=f'{int(PERCENTILE * 100)}th Percentile Cutoff ({threshold:.1f})')
    plt.title("Filtering Artifacts via GMM Percentile")
    plt.xlabel("Filament Length")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
