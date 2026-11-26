"""
Add all scripts that can be used np matter the step in the pipeline like
napari or visualization methods..

- should store nrrd or mrc be specific to the step or not?
"""
import math, matplotlib, napari, pyvis, os, nrrd, mrcfile
import numpy as np
import nibabel as nib
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from jinja2 import Environment, FileSystemLoader
matplotlib.use('TkAgg')  # or 'QtAgg'  # otherwise an error when plotting each slice


def show_all_slices(volume):
    """ Visualization of all the slices of the 3D embedded object """
    # !!! Implemented to handle data in format (Z, Y, X) => may need to transpose (.T) when calling
    num_slices = volume.shape[0]

    # Define grid size (square-ish)
    cols = math.ceil(math.sqrt(num_slices))
    rows = math.ceil(num_slices / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

    # Flatten axes for easy iteration
    axes = np.array(axes).reshape(-1)

    for i in range(num_slices):
        axes[i].imshow(volume[i], cmap='gray')
        axes[i].set_title(f"z={i}", fontsize=16)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_slices, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def view_in_napari(colors=None, point_sizes=None, **kwargs):
    """
    Generalize a napari visualization.
    Kwargs should be dictionaries (one for each layer) mapping layer names to data:
    image={"vol": vol_array, "skel": skel_array}
    """
    viewer = napari.Viewer()
    colors = colors or []
    point_sizes = point_sizes or []

    color_idx = 0
    size_idx = 0

    for layer_type in ["image", "labels", "points"]:
        if layer_type not in kwargs:
            continue

        layer_data = kwargs[layer_type]

        # If a list is passed instead of a dict, auto-generate safe names
        if isinstance(layer_data, list):
            layer_data = {f"{layer_type}_{i}": arr for i, arr in enumerate(layer_data)}

        for var_name, val in layer_data.items():
            layer_props = {"name": var_name}

            # Add Color
            if color_idx < len(colors) and colors[color_idx] is not None:
                if layer_type == "points":
                    layer_props["face_color"] = colors[color_idx]
                else:
                    layer_props["colormap"] = colors[color_idx]

            # Add Size (Points only)
            if layer_type == "points" and size_idx < len(point_sizes):
                layer_props["size"] = point_sizes[size_idx]
                size_idx += 1

            # Transformation from XYZ to ZYX - necessary as Napari expects data in (Z, Y, X)
            if layer_type in ["image", "labels"]:
                # Transpose the 3D array: (X, Y, Z) -> (Z, Y, X)
                display_val = np.transpose(val, (2, 1, 0))
            elif layer_type == "points":
                # Convert list/array to a numpy array first
                points_arr = np.array(val)

                # Check if the points array is empty to prevent slicing errors
                if points_arr.size > 0:
                    # Reverse the columns: [x, y, z] -> [z, y, x]
                    display_val = points_arr[:, ::-1]
                else:
                    display_val = points_arr

            # Add the layer
            if layer_type == "image":
                viewer.add_image(display_val, **layer_props)
            elif layer_type == "labels":
                viewer.add_labels(display_val, **layer_props)
            elif layer_type == "points":
                viewer.add_points(display_val, **layer_props)

            color_idx += 1

    napari.run()


def html_vis(graph: nx.Graph, output_path: str):
    """ Generate and store html visualization of graph """
    net = Network(height="800px", width="100%", directed=False, notebook=False)

    # ---- Manually load PyVis template (your workaround) ----
    template_path = os.path.join(os.path.dirname(pyvis.__file__), "templates")
    net.templateEnv = Environment(loader=FileSystemLoader(template_path))
    net.template = net.templateEnv.get_template("template.html")

    net.from_nx(graph)

    net.show_buttons()
    net.toggle_physics(True)

    net.show(output_path)


def mask_picks_nii(graph_rep, candidates, original_nii_path, output_nii_path, radius):
    """
    Clones the header and shape of the original mask, draws candidates
    directly in (X, Y, Z) space, and exports them perfectly aligned.
    """
    if radius is None:
        radius = 10

    original = nib.load(original_nii_path)

    # Create a blank mask natively in the (X, Y, Z) layout
    new_mask_xyz = np.zeros(original.shape, dtype=np.uint8)

    # Mark the candidates
    for node in candidates:
        if graph_rep is None:
            # Unpack in (X, Y, Z)
            x, y, z = int(node[0]), int(node[1]), int(node[2])
        else:
            coords = graph_rep.nodes[node]['coords']
            x, y, z = int(coords[0]), int(coords[1]), int(coords[2])

        x_min = max(0, x - radius)
        x_max = min(new_mask_xyz.shape[0], x + radius + 1)

        y_min = max(0, y - radius)
        y_max = min(new_mask_xyz.shape[1], y + radius + 1)

        z_min = max(0, z - radius)
        z_max = min(new_mask_xyz.shape[2], z + radius + 1)

        # Decide filling approach
        # 1. Full 3D Box
        new_mask_xyz[x_min:x_max, y_min:y_max, z_min:z_max] = 1

        # 2. A flat bounding box on a single Z-slice (Axial view)
        # new_mask_xyz[x_min:x_max, y_min:y_max, z] = 1

    # Save using the original header
    new_image = nib.Nifti1Image(new_mask_xyz, original.affine, header=original.header.copy())
    nib.save(new_image, output_nii_path)


def mask_picks_nii(graph_rep, candidates, original_nii_path, output_nii_path, radius):
    """
    Clones the header and shape of the original mask, draws candidates
    directly in (X, Y, Z) space, and exports them perfectly aligned.
    """
    if radius is None:
        radius = 10

    original = nib.load(original_nii_path)

    # Create a blank mask natively in the (X, Y, Z) layout
    new_mask_xyz = np.zeros(original.shape, dtype=np.uint8)

    # Mark the candidates
    for node in candidates:
        if graph_rep is None:
            # Unpack in (X, Y, Z)
            x, y, z = int(node[0]), int(node[1]), int(node[2])
        else:
            coords = graph_rep.nodes[node]['coords']
            x, y, z = int(coords[0]), int(coords[1]), int(coords[2])

        x_min = max(0, x - radius)
        x_max = min(new_mask_xyz.shape[0], x + radius + 1)

        y_min = max(0, y - radius)
        y_max = min(new_mask_xyz.shape[1], y + radius + 1)

        z_min = max(0, z - radius)
        z_max = min(new_mask_xyz.shape[2], z + radius + 1)

        # Decide filling approach
        # 1. Full 3D Box
        new_mask_xyz[x_min:x_max, y_min:y_max, z_min:z_max] = 1

        # 2. A flat bounding box on a single Z-slice (Axial view)
        # new_mask_xyz[x_min:x_max, y_min:y_max, z] = 1

    # Save using the original header
    new_image = nib.Nifti1Image(new_mask_xyz, original.affine, header=original.header.copy())
    nib.save(new_image, output_nii_path)


def mask_picks_nrrd(graph_rep, candidates, original_nrrd_path, output_nrrd_path, radius=10):
    """
    Clones the header and shape of the original mask, draws candidates
    directly in (X, Y, Z) space, and exports them perfectly aligned.
    """
    if radius is None:
        radius = 10

    original, header = nrrd.read(original_nrrd_path)

    # Create a blank mask natively in the (X, Y, Z) layout
    new_mask_xyz = np.zeros(original.shape, dtype=np.uint8)

    # Mark the candidates
    for node in candidates:
        if graph_rep is None:
            # Unpack in (X, Y, Z)
            x, y, z = int(node[0]), int(node[1]), int(node[2])
        else:
            coords = graph_rep.nodes[node]['coords']
            x, y, z = int(coords[0]), int(coords[1]), int(coords[2])

        x_min = max(0, x - radius)
        x_max = min(new_mask_xyz.shape[0], x + radius + 1)

        y_min = max(0, y - radius)
        y_max = min(new_mask_xyz.shape[1], y + radius + 1)

        z_min = max(0, z - radius)
        z_max = min(new_mask_xyz.shape[2], z + radius + 1)

        # Decide filling approach
        # 1. Full 3D Box
        new_mask_xyz[x_min:x_max, y_min:y_max, z_min:z_max] = 1

        # 2. A flat bounding box on a single Z-slice (Axial view)
        # new_mask_xyz[x_min:x_max, y_min:y_max, z] = 1

    # Save using the original header
    nrrd.write(output_nrrd_path, new_mask_xyz, header)


def export_arp23_nrrd_from_template(graph, arp23_candidates, template_nrrd_path, output_nrrd_path, radius=10):
    """
    Clones the header and shape of the original NRRD mask, draws candidates
    in Python-native (Z, Y, X) space, and exports as MITK-native (X, Y, Z).
    """

    # Read the NRRD mask (loads as X, Y, Z)
    header = nrrd.read_header(template_nrrd_path)
    x_template_size, y_template_size, z_template_size = header['sizes']

    # Create a blank mask in the (Z, Y, X) layout
    new_mask_xyz = np.zeros((z_template_size, y_template_size, x_template_size), dtype=np.uint8)

    # Mark the candidates
    for node in arp23_candidates:
        coord = graph.nodes[node]['coords']

        x, y, z = int(coord[0]), int(coord[1]), int(coord[2])

        x_min = max(0, x - radius)
        x_max = min(new_mask_xyz.shape[0], x + radius + 1)

        y_min = max(0, y - radius)
        y_max = min(new_mask_xyz.shape[1], y + radius + 1)

        z_min = max(0, z - radius)
        z_max = min(new_mask_xyz.shape[2], z + radius + 1)

        # Full 3D Box
        # new_mask_xyz[x_min:x_max, y_min:y_max, z_min:z_max] = 1

        # A flat bounding box on a single Z-slice
        new_mask_xyz[x_min:x_max, y_min:y_max, z] = 1

    # 5. Save using the EXACT original header
    nrrd.write(output_nrrd_path, new_mask_xyz, header)


def store_generated_volume(volume: np.ndarray, output_filepath: str, header: str = None):
    """ Write the array and metadata to an .nrrd file """
    try:
        nrrd.write(output_filepath, volume.astype(np.uint8), header=header)
    except Exception as e:
        raise RuntimeError(f"Failed to save NRRD file: {e}")


def store_skeleton_mrc(vol_skeleton: np.ndarray, output_filepath: str):
    """ Store the array in an .mrc file """
    with mrcfile.new(output_filepath, overwrite=True) as mrc:
        # Convert ndarray from boolean to numeric and save
        mrc.set_data(vol_skeleton.astype(np.int8))
