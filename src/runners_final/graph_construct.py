import argparse
import pickle
from pathlib import Path
from skimage import morphology
from typing import Optional, Tuple

from src.pipeline.graph_constructor import *
from src.pipeline.mask_processing import *
from src.pipeline.properties_computation.angles_at_intersections import *
from filepath_transformer import *


def edit_filepaths(input_path: str, output_path: str, html_output: Optional[str]) -> Tuple[str, str, Optional[str]]:
    final_input = transform_filepath(input_path, DEFAULT_INPUT_PATH, must_exist=True)
    final_output = transform_filepath(output_path, DEFAULT_OUTPUT_PATH, must_exist=False)
    final_html = transform_filepath(html_output, DEFAULT_HTML_PATH, must_exist=False)
    return final_input, final_output, final_html


if __name__ == "__main__":
    # Read and process commandline arguments
    parser = argparse.ArgumentParser("Generate graph representation")
    parser.add_argument("-i", "--input", required=True, type=str, help="Input binary segmentation to process.")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output path to store the generated graph.")
    parser.add_argument("-oh", "--output_html", type=str, help="File path to store html representation of graph.")

    # Additional parameters:
    parser.add_argument("-cr", "--closing_radius", type=int, help="Specify the radius of the ball used in closing.")

    args = parser.parse_args()

    input_data_path, graph_store_path, html_path = edit_filepaths(args.input, args.output, args.output_html)

    if input_data_path.endswith(".nii"):
        original_nii = nib.load(input_data_path)
        mask = original_nii.get_fdata()

        # Extract and store voxel size
        voxel_size = original_nii.header.get_zooms()  # Extract Voxel Size (spacing in X, Y, Z)
    elif input_data_path.endswith(".nrrd"):
        mask, header = nrrd.read(input_data_path)

        # Extract and store voxel size
        voxel_size = header.get('spacings', header.get('space directions'))
    else:
        raise Exception("Unknown file type! Currently handled input file types are .nii and .nrrd")

    # -----------------------------------------------------------------

    print("Morphological Closing Operation:")
    mask = morphology.closing(mask, footprint=morphology.ball(
        radius=int(args.closing_radius) if args.closing_radius is not None else 2))

    print("Skeletonization:")
    mask_skeleton = skeletonization(mask)

    print("Skeleton Voxel Classification:")
    intersections_nc, end_positions, final_intersections, intersection_map = local_test_nc(mask_skeleton)

    print("Graph Construction:")
    graph_representation = graph_construction(mask_skeleton, np.argwhere(end_positions), final_intersections,
                                              intersections_nc, mask, intersection_map, voxel_size)
    # Store results
    # Avoid double '.pkl.pkl' if the user already included the extension
    if not graph_store_path.endswith(".pkl"):
        graph_store_path = f"{graph_store_path}.pkl"

    with open(graph_store_path, "wb") as f:
        pickle.dump(graph_representation, f)

    if html_path is not None:
        html_vis(graph=graph_representation, output_path=html_path)

