import argparse, pickle, time
from src.pipeline.properties_computation.angles_at_intersections import *
from src.pipeline.utils_vis.vis import *

ANGLE_SIGNATURE = (70, 110, 180)
TOLERANCE = 10


if __name__ == "__main__":
    # Read and process commandline arguments
    parser = argparse.ArgumentParser("Geometrical analysis of graph")
    parser.add_argument("-i", "--input", required=True, type=str, help="Graph storage path")
    # ToDo: parser for lower limit for straight section after intersection
    # ToDo: parser for upper limit for straight section after intersection
    # ToDo: add argument for output of results mask
    # ToDo: add input path to extract header information

    args = parser.parse_args()

    # Load stored graph
    graph_path = args.input
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    # Compute angles at each intersection
    compute_dirs_and_angles(G, 5, 40)

    # Filter all intersection
    arp23_candidates = get_intersection_candidates(G, ANGLE_SIGNATURE, TOLERANCE)

    # Store picks on a mask for visual inspection in MITK
    export_arp23_nrrd_from_template(G, arp23_candidates,
                                    template_nrrd_path="D:/BA_data/podosomes/108_0000.nrrd",
                                    output_nrrd_path="/data/generated_data/other/arp23_mask_podosome.nrrd",
                                    radius=10)

    #export_arp23_nrrd_from_template(G, arp23_candidates,
    #                                template_nrrd_path="C:/Users/ant/PycharmProjects/bachelor_thesis/data/segmentation_masks/actin_Pos_28_2.nrrd",
    #                                output_nrrd_path="C:/Users/ant/PycharmProjects/bachelor_thesis/data/generated_data/other/arp23_mask_synapse_test.nrrd",
    #                                radius=10)
