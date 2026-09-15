import nrrd, pickle
import numpy as np
from skimage import morphology
from src.pipeline.mask_processing import *
from src.pipeline.graph_constructor import *
from src.pipeline.properties_computation.angles_at_intersections import *

"""
Split actin segmentation based on labeling mask. 
If 
"""

GENERAL_PATH = "C:/Users/ant/PycharmProjects/bachelor_thesis/data/segmentation_masks/"
STORE_MASK = False
STORE_GRAPH = False
GRAPHS_PATH = "C:/Users/ant/PycharmProjects/bachelor_thesis/data/generated_data/graphs/"
VOXEL_SIZE = 10  # Can be extracted also from the nrrd header


def plot_distribution_distances(list_distances, title, x_axis_label: str = None):
    plt.figure(figsize=(14, 6))

    min_dist = int(np.floor(min(list_distances)))
    max_dist = int(np.ceil(max(list_distances)))

    bin_edges = np.linspace(min_dist, max_dist, 10)
    bin_edges = np.unique(np.round(bin_edges).astype(int))

    # Calculate histogram
    counts, bin_edges = np.histogram(list_distances, bins=bin_edges)

    # Equal-width categorical positions for the bars
    x_positions = np.arange(len(counts))

    # Plot bars
    bars = plt.bar(x_positions, counts, width=0.9, color='royalblue', edgecolor='black', alpha=0.8)

    # X-axis labels
    labels = []

    for i in range(len(counts)):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        # Label each bar with the actual range it represents
        labels.append(f"{left}-{right}")

    plt.xticks(x_positions, labels, rotation=45, fontsize=9)
    plt.xlim(-0.5, len(counts) - 0.5)

    # Add count above each bar
    for bar in bars:
        height = bar.get_height()

        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom',
                fontsize=9, color='black')

    # Labels, title, and grid
    plt.title(title, fontsize=14)
    plt.xlabel(x_axis_label, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    actin_mask, header_a = nrrd.read(f'{GENERAL_PATH}actin_Pos_28_2.nrrd')
    t_cell_mask, header_t = nrrd.read(f'{GENERAL_PATH}line.nrrd')

    actin_mask = actin_mask.astype(bool)
    t_cell_mask = t_cell_mask.astype(bool)

    actin_t_cell = actin_mask & t_cell_mask
    actin_cancer_cell = actin_mask & (~t_cell_mask)

    if STORE_MASK:
        actin_t_cell = actin_t_cell.astype(np.uint8)
        actin_cancer_cell = actin_cancer_cell.astype(np.uint8)

        nrrd.write(f"{GENERAL_PATH}actin_t_cell.nrrd", actin_t_cell, header_a)
        nrrd.write(f"{GENERAL_PATH}actin_cancer_cell.nrrd", actin_cancer_cell, header_a)

    # Actin has been split, how run independently
    if STORE_GRAPH:
        for mask, cell in zip([actin_t_cell, actin_cancer_cell], ["t_cell", "cancer"]):
            # Add voxel_size
            print("Morphological Closing Operation:")
            mask = morphology.closing(mask, footprint=morphology.ball(radius=2))

            print("Skeletonization:")
            mask_skeleton = skeletonization(mask)

            print("Local Test (Find Intersections and End Points):")
            intersections_nc, end_positions, final_intersections, intersection_map = local_test_nc(mask_skeleton)

            print("Graph Construction:")
            graph_representation = graph_construction(mask_skeleton, np.argwhere(end_positions), final_intersections,
                                                      intersections_nc, mask, intersection_map)

            print("Store graph")
            with open(f"{GRAPHS_PATH}graph_actin_{cell}_.pkl", "wb") as f:
                pickle.dump(graph_representation, f)

            print("Store html")
            html_vis(graph=graph_representation,
                     output_path=f"C:/Users/ant/PycharmProjects/bachelor_thesis/data/generated_data/htmls/html_graph_{cell}.html")

    # Once stored, load and analyze
    with open(f"{GRAPHS_PATH}graph_actin_t_cell_.pkl", "rb") as f:
        graph_rep_t_cell = pickle.load(f)

    with open(f"{GRAPHS_PATH}graph_actin_cancer_.pkl", "rb") as f:
        graph_rep_cancer = pickle.load(f)

    for G, title in zip([graph_rep_t_cell, graph_rep_cancer], ["T", "Cancer"]):
        # Get intersections
        intersection_nodes = [
            node for node, data in G.nodes(data=True)
            if data.get("node_type") == 'intersection']

        print("Number of intersection nodes: ", len(intersection_nodes))

        # Get grad of all intersection
        intersection_grad = [len(G.nodes[node_id]['connections'].keys()) for node_id in intersection_nodes]
        
        print("Intersection grad:")
        print(intersection_grad)
        print(f"Averaged intersection grad "
              f"{0 if len(intersection_grad) == 0 else sum(intersection_grad)/len(intersection_grad)}")

        # Plot histogram with that
        # 1. Dynamically find the start and end based on your data
        min_val = min(intersection_grad)
        max_val = max(intersection_grad)

        if min_val == max_val:
            min_val = min_val - 1
            max_val = max_val + 1

        # 2. Create the histogram and capture the output
        # bins=range(...) ensures exactly one bin per integer.
        # align='left' centers the bar directly over the integer tick on the x-axis.
        counts, bins, patches = plt.hist(
            intersection_grad,
            bins=range(min_val, max_val + 2),  # +2 ensures the final integer is fully included in the range
            align='left',
            color='skyblue',
            edgecolor='black'
        )

        # 3. Add the exact count on top of each bar
        # We use 'patches', which contains the actual rectangle shapes drawn by plt.hist
        plt.bar_label(patches, padding=3)

        # 4. Format the X-axis so it only shows exact integers
        plt.xticks(range(min_val, max_val + 1))
        plt.ylim(top=max(counts) * 1.10)

        # Labels and titles
        plt.xlabel('Intersection Degree')
        plt.ylabel('Frequency')
        plt.title(f'Intersection Degree for Actin Network in the {title} Cell')

        # Keep things neat and save
        plt.tight_layout()
        plt.savefig(f'C:/Users/ant/PycharmProjects/bachelor_thesis/data/generated_data/report/inters_grad_hist_{title}_2.jpg')
        plt.show()



        # --------------------    Filaments
        print(f"Number of all filaments {G.graph['all_filaments']}")
        all_filament_lengths = [value.length()*VOXEL_SIZE for key, value in G.graph["filament_paths"].items()]

        # Find the max value to know where to stop the x-axis
        max_len = max(all_filament_lengths)

        # Create bins every 150 Ångström (from 0 up to max_len + 150)
        my_bins = np.arange(0, max_len + 150, 150)

        plt.figure(figsize=(8, 5))
        counts, bins, patches = plt.hist(
            all_filament_lengths,
            bins=my_bins,
            color='#4C72B0',
            edgecolor='black'
        )

        # Optional: Still add the labels on top for clarity
        plt.bar_label(patches, padding=3)
        plt.xticks(my_bins, rotation=45, ha='right')
        plt.ylim(top=max(counts) * 1.10)

        plt.xlabel('Subfilament Length in Ångström')
        plt.ylabel('Frequency')
        plt.title(f'Subfilament Length for Actin Network in the {title} Cell')

        plt.tight_layout()
        plt.savefig(
            f'C:/Users/ant/PycharmProjects/bachelor_thesis/data/generated_data/report/fil_len_hist_{title}_2.jpg')

        plt.show()

        plot_distribution_distances(all_filament_lengths, f"Distribution of All Subfilament Lengths in {title} Cells", "Subfilament Length in Ångström")
        print("Go to next graph")

