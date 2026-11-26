import pickle, argparse
import pandas as pd

from filepath_transformer import *
from src.pipeline.properties_computation.angles_at_intersections import *

THRESHOLD = 20
LOWER_LIMIT = 10
UPPER_LIMIT = 35
ANGLE_SIGNATURE = (70, 110, 180)


if __name__ == "__main__":
    # Read and process commandline arguments
    parser = argparse.ArgumentParser("Run graph analysis")
    parser.add_argument("-i", "--input", required=True, type=str, help="Input graph to process.")
    parser.add_argument("-im", "--input_mask", required=True, type=str,
                        help="Filepath to segmentation mask used to generate the graph")
    parser.add_argument("-m", "--output_mask_filepath", required=True, type=str,
                        help="Where to store the generated mask with predicted Arp2/3 picks")

    # Additional arguments
    parser.add_argument("-t", "--threshold", type=int, help="Give degree of freedom.")
    parser.add_argument("-ll", "--lower_limit", type=int, help="Give from which voxel to calculate the direction.")
    parser.add_argument("-ul", "--upper_limit", type=int,
                        help="Give until which voxel to consider for the direction calculation.")
    parser.add_argument("-r", "--radius", type=int,
                        help="Give radius of the sphere used to mark predicted intersections in the generated mask.")

    args = parser.parse_args()

    graph_input = transform_filepath(args.input, DEFAULT_INPUT_PATH, must_exist=True)
    original_input_mask = transform_filepath(args.input_mask, DEFAULT_INPUT_PATH, must_exist=False)
    output_mask_filepath = transform_filepath(args.output_mask_filepath, DEFAULT_INPUT_PATH, must_exist=False)
    # take the name of the file and add _degree.png OR _subfil_len.png
    general_plot_path = DEFAULT_PLOT_PATH / Path(graph_input).stem

    nii = True if original_input_mask.endswith(".nii") else False

    with open(graph_input, "rb") as f:
        graph_representation = pickle.load(f)

    print("Angles at Intersections:")
    compute_dirs_and_angles(graph=graph_representation,
                            lower_limit=int(args.lower_limit) if args.lower_limit is not None else LOWER_LIMIT,
                            upper_limit=int(args.upper_limit) if args.upper_limit is not None else UPPER_LIMIT)

    print("Filter Intersection Candidates:")
    # Select Arp2/3 candidate nodes based on angle criteria
    arp23_candidates = get_intersection_candidates(graph=graph_representation,
                                                   target_angles=ANGLE_SIGNATURE,
                                                   threshold=int(args.threshold) if args.threshold is not None else THRESHOLD)

    # Generate a mask from predicted Arp2/3 picks
    if nii:
        mask_picks_nii(graph_rep=graph_representation,
                       candidates=arp23_candidates,
                       original_nii_path=original_input_mask,
                       output_nii_path=output_mask_filepath,
                       radius=int(args.radius) if args.radius is not None else None)
    else:
        mask_picks_nrrd(graph_rep=graph_representation,
                        candidates=arp23_candidates,
                        original_nrrd_path=original_input_mask,
                        output_nrrd_path=output_mask_filepath,
                        radius=int(args.radius) if args.radius is not None else None)

    # Collect subfilament lengths and intersection degrees for plotting
    all_fil_lengths = []  # ToDo: NOT CORRECT!!! - here only filaments at intersections; but there are filaments at start positions and not going to an intersection
    all_intersection_degrees = []
    fil_check = set()
    for node, attr in graph_representation.nodes(data=True):
        if attr.get('node_type') == "intersection":  # get all intersection nodes
            # Get degree of the intersection node
            all_intersection_degrees.append(len(attr.get("connections")))

    for node, attr in graph_representation.nodes(data=True):
        if attr.get('node_type') == "intersection":  # get all intersection nodes
            # Collect the lengths of the filaments
            for fil in attr.get("connections"):
                if fil not in fil_check:  # Make sure to not add the length of a subfilament twice
                    all_fil_lengths.append(graph_representation.graph["filament_paths"][fil].length())
                    fil_check.add(fil)
        else:
            if attr.get("fil_num") not in fil_check:
                all_fil_lengths.append(graph_representation.graph["filament_paths"][attr.get("fil_num")].length())
                fil_check.add(attr.get("fil_num"))

    # plot
    # --> intersection degree
    plot_len_degree(all_intersection_degrees, "Distribution of Intersection Degrees",
                    f"{general_plot_path}_degrees.png", "Degree")

    # --> subfil length
    plot_len_degree(all_fil_lengths, "Distribution of Subfilament Lengths",
                    f"{general_plot_path}_subfil_len.png", "Length in Voxels")

    # use pandas to print on the console basic statistics
    print(f"Basic statistics about the degree of all intersections: \n{pd.Series(all_intersection_degrees).describe()}")
    print(f"Basic statistics about all subfilament lengths in voxels: \n{pd.Series(all_fil_lengths).describe()}")

