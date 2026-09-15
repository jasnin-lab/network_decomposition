import argparse, time, sys
from skimage import morphology
from src.pipeline.mask_processing import *
from src.pipeline.graph_constructor import *
from src.pipeline.synthetic_dataset_generation import *


def create_store_mask(original_mask: np.ndarray, intersections: np.ndarray):
    intersections_mask = np.zeros_like(original_mask)

    for inters_coord in intersections:
        # Unpack explicitly as Z, Y, X to avoid index confusion!
        z_c, y_c, x_c = inters_coord[0], inters_coord[1], inters_coord[2]

        # Convert nm radius in voxel units along each axis
        rx = ry = rz = 10

        # Define bounding box around the ribosome center
        # X uses x_c and shape[2]
        x_min = max(0, int(np.floor(x_c - rx)))
        x_max = min(intersections_mask.shape[2] - 1, int(np.ceil(x_c + rx)))

        # Y uses y_c and shape[1]
        y_min = max(0, int(np.floor(y_c - ry)))
        y_max = min(intersections_mask.shape[1] - 1, int(np.ceil(y_c + ry)))

        # Z uses z_c and shape[0]
        z_min = max(0, int(np.floor(z_c - rz)))
        z_max = min(intersections_mask.shape[0] - 1, int(np.ceil(z_c + rz)))

        # Create coordinate grids
        z, y, x = np.ogrid[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]

        # Compute distances (matching axes to their respective centers)
        dist_z = (z - z_c)
        dist_y = (y - y_c)
        dist_x = (x - x_c)

        # Compare squared distance against squared radius (100)
        squared_distance = (dist_x ** 2 + dist_y ** 2 + dist_z ** 2)

        # Select voxels inside the ribosome sphere
        inside = squared_distance <= 100

        # Label the voxels in the segmentation mask
        intersections_mask[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1][inside] = 1

    # loop done => visualize in napari and store as mrc
    print(f"labeled voxels before saving the mrc {np.count_nonzero(intersections_mask)}")
    # vis_napari(original_mask, intersections_mask)

    mitk_array = np.transpose(intersections_mask, (2, 1, 0))
    mitk_array = mitk_array.astype(np.int8)

    # 3. Create the new MRC file and inject the metadata
    with mrcfile.new("intersections_mask_mrc.mrc", overwrite=True) as mrc:
        mrc.set_data(mitk_array)

        # Apply voxel size
        # (Using float32 to ensure compatibility with MRC spec)
        mrc.voxel_size = (
            np.float32(voxel_size[0]),
            np.float32(voxel_size[1]),
            np.float32(voxel_size[2])
        )

        # Apply physical origin (where the volume sits in space)
        # Note: mrcfile allows direct assignment to these sub-fields
        mrc.header.origin.x = np.float32(affine[0, 3])
        mrc.header.origin.y = np.float32(affine[1, 3])
        mrc.header.origin.z = np.float32(affine[2, 3])

    print("Successfully generated MRC file for intersection points")


if __name__ == "__main__":
    # Read and process commandline arguments
    parser = argparse.ArgumentParser("Generate graph representation")
    parser.add_argument("-i", "--input", required=False, type=str, help="Input binary segmentation to process")
    parser.add_argument("-v", "--volume", action="store_true", help="Provide if the input is already a volume")

    # Possible (non-obligatory) hyperparameters for:
    # --> Synthetic data generation
    parser.add_argument("-sr", "--sphere_radius", type=int, help="Sphere radius")  # ToDo: could restrict datatype
    parser.add_argument("-du", "--dilation_uniform", action="store_true",
                        help="Provide tag if the sphere dilation should not be uniform")  # Then a provided tag = False.
    parser.add_argument("-t", "--tilt", action="store_true",
                        help="Provide if the synthetic instance should be 90° tilted")
    parser.add_argument("-sg", "--store_generated_vol", type=str, help="Output file path to store generated volume")
    # --> Volume processing
    parser.add_argument("-ss", "--store_skeleton", type=str, help="Output path to store the skeleton of the 3D volume")
    # ToDo: add to store graph path
    # ToDo: add to store graph visualization
    # ToDo: add other hyperparameters...

    args = parser.parse_args()

    data_path = args.input
    nii = False  # ToDo: change!!!
    voxel_size = None  # not relevant for synthetic dataset
    print(args.dilation_uniform)
    # ToDo: check if path exists - handle possible errors -> or move to check_input_type function
    if not args.volume:  # If volume tag is not provided => generate a synthetic volume
        # Generate 3D volume from 2D volume
        mask = generate_3d_synthetic_volume(image_path=data_path, uniform_dilation=args.dilation_uniform, sphere_radius=7,
                                            mode=1, tilt=args.tilt, image_binarize_threshold=127)
        # ToDo: do something for the number-parameters in the function call

        # Store generated volume if output path is provided
        if args.store_generated_vol is not None:
            store_generated_volume(mask, args.store_generated_vol)  # ToDo: change!!!!
    else:  # expect .nrrd and .nii as format for volumetric data
        file_type = ""
        if data_path.endswith(".nii"):
            original_nii = nib.load(data_path)
            mask = original_nii.get_fdata()
            affine = original_nii.affine
            file_type = ".nii"

            # Extract and store voxel size
            voxel_size = original_nii.header.get_zooms()  # Extract Voxel Size (spacing in X, Y, Z)

        elif data_path.endswith(".nrrd"):
            mask, header = nrrd.read(data_path)
            file_type = ".nrrd"

            # Extract and store voxel size
            voxel_size = header.get('spacings', header.get('space directions'))
            # the get() function can use an alternative if the first option is not provided

        else:
            raise Exception("Unknown file type! Currently handled input file types are .nii and .nrrd")

    # -------------------------------------------- PIPELINE --------------------------------------------
    # If a synthetic volume was generated, view it
    # show_all_slices(mask.T)

    # Step 1: Skeletonization
    print("Apply closing morphological operation and skeletonize")
    start_time = time.perf_counter()
    mask = morphology.closing(mask, footprint=morphology.ball(radius=2))
    end_time = time.perf_counter()
    print(f"Morphological closing done in {(end_time - start_time):.6f}s")

    # show_all_slices(mask.T)  # ToDo: make in function .T - to not have to do this here!!!
    print("Now Skeletonization:")
    start_time = time.perf_counter()
    mask_skeleton = skeletonization(mask)
    end_time = time.perf_counter()
    print(f"Skeletonization done in {(end_time - start_time):.6f}s")

    print("View skeleton in napari")
    # view_in_napari(colors=['blue', 'white'], point_sizes=None, image={"Mask": mask, "Skeleton": mask_skeleton})

    # Import skeleton storage function, make it general for input data type and use input filepath to assign metadata
    # Store skeleton to visualize  # ToDo: make a general, utils function
    """
    skeleton_to_store = np.transpose(mask_skeleton, (2, 1, 0)).astype(np.uint8)

    if "header" not in locals():  # StackOverflow
        header = nrrd.read_header("D:/BA_data/podosomes/108_0000.nrrd")

    nrrd.write("/data/generated_data/other/skeleton_podosome.nrrd",
               skeleton_to_store, header)

    print("skeleton_saved!!")
    """

    # Step 2: Local test - NC
    print("Second step - Find Intersections and End points by performing the Local test")
    start_time = time.perf_counter()
    intersections_nc, end_positions, final_intersections, intersection_map = local_test_nc(mask_skeleton)
    end_time = time.perf_counter()
    print(f"Local Test done in {(end_time - start_time):.6f}s")

    """
    view_in_napari(colors=['white', 'blue', None, None, 'white'], point_sizes=[1],
                   image={"Segmentation": mask, "Skeleton": mask_skeleton},
                   labels={"Potential Intersections": intersections_nc, "End points": end_positions},
                   points={"Mean-Shifted Intersections": final_intersections})
    """


    # ToDo: make store intersections function and create a mask to visualize result - based on input data type
    # final_intersection - ndarray
    #if nii:
    #    create_store_mask(mask, final_intersections)

    # print("Now test out area based")  # - I would keep as an example testing workflow that can be run
    # intersections_ab = test_triangle_area(mask_skeleton)
    # view_in_napari(mask, mask_skeleton, intersections_ab)

    print("Graph Construction")
    start_time = time.perf_counter()
    graph_representation = graph_construction(mask_skeleton, np.argwhere(end_positions), final_intersections,
                                              intersections_nc, mask, intersection_map, voxel_size)
    end_time = time.perf_counter()
    print(f"Graph construction runtime: {(end_time - start_time):.6f}s")

    # Store generated graphs to speed-up analysis
    # To avoid generating them again from scratch
    # Update them only if necessary
    # ToDo: make input name-dynamic

    #with open("/data/generated_data/graphs/graph_podosome.pkl","wb") as f:
    #    pickle.dump(graph_representation, f)

    """
    view_in_napari(colors=['white', 'white', None, None, 'white'], point_sizes=[1],
                   image={"Segmentation": mask, "Skeleton": mask_skeleton},
                   labels={"Potential Intersections": intersections_nc, "End points": end_positions},
                   points={"Mean-Shifted Intersections": final_intersections})
    """

    intersection_nodes = [
        node for node, data in graph_representation.nodes(data=True)
        if data.get("node_type") == 'intersection']

    print("Number of intersection nodes: ", len(intersection_nodes))

    # Get grad of all intersection
    intersection_grad = [len(graph_representation.nodes[node_id]['connections'].keys()) for node_id in intersection_nodes]
    print("Intersection grad:")
    print(intersection_grad)
    print(f"Averaged intersection grad "
          f"{0 if len(intersection_grad) == 0 else sum(intersection_grad) / len(intersection_grad)}")


    # html_vis(graph=graph_representation, output_path="/data/graphs_plots/html_graph_star_12_normal.html")
