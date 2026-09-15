"""
The following script generates instances for the synthetic dataset by taking a 2D image (a letter),
skeletonizing it in 2D and embedding the resulting skeleton in 3D. The image can be tilted to 90°
if the corresponding parameter is provided
Usage: ...
"""

# Import modules
import nrrd, napari
import numpy as np
from PIL import Image
from skimage import morphology
from scipy.ndimage import rotate, binary_dilation
from skimage.morphology import skeletonize, ball, medial_axis, thin
from src.pipeline.utils_vis import *
from src.pipeline.mask_processing import *
from src.pipeline.graph_constructor import *


# Hyperparameters and default values
LETTER = "T"  # Letters to check: O, Q, T, R, Y, P, S, B, X, L (jpeg); J_BG; CHI
INPUT_PATH = f"C:/Users/ant/PycharmProjects/bachelor_thesis/data/input_data/{LETTER}_simple.jpg"  # ToDo: make path relative, not absolute; to make it not only locally runnable
EMPTY_SLICES = 3  # Usually int(EMPTY_SLICES/2) per side (left/right from 3D dilated object)
SPHERE_RADIUS = 3
TILT = False
UNIFORM_DILATION = False

# ToDo: make mode + image_binarize_threshold: - global variables and reduce the number of parameters for generation!!!!!

if TILT:
    OUTPUT_PATH = f"C:/Users/ant/PycharmProjects/bachelor_thesis/data/generated_data/synthetic_dataset/{LETTER}_90tilted_volume.nrrd"
else:
    OUTPUT_PATH = f"C:/Users/ant/PycharmProjects/bachelor_thesis/data/generated_data/synthetic_dataset/{LETTER}_volume.nrrd"


def vis_result(bin_array: np.ndarray):
    """ Visualization of 2D binary array as an image """
    vis_array = (bin_array * 255).astype(np.uint8)
    Image.fromarray(vis_array).show()
    # Image.fromarray(vis_array).save("C:/Users/ant/PycharmProjects/bachelor_thesis/data/generated_data/report/2d_skel.jpg")


def non_uniform_dilation(skeleton_mask: np.ndarray, radii_choices: list):
    # Generate ball structuring element for each provided radii
    balls = {r: ball(r).astype(bool) for r in radii_choices}

    # Generate result array and extract skeletal voxel coordinates
    dilated_volume = np.zeros_like(skeleton_mask, dtype=np.uint8)
    skeleton_coords = np.argwhere(skeleton_mask > 0)

    Z, Y, X = dilated_volume.shape

    for coord in skeleton_coords:
        z_c, y_c, x_c = coord[0], coord[1], coord[2]
        r = np.random.choice(radii_choices)
        selected_ball = balls[r]

        # Target Array Bounds (preventing negative indices and overflow)
        z_min, z_max = max(0, z_c - r), min(Z, z_c + r + 1)
        y_min, y_max = max(0, y_c - r), min(Y, y_c + r + 1)
        x_min, x_max = max(0, x_c - r), min(X, x_c + r + 1)

        # Ball Bounds (if the array gets cut, we cut the ball mask too)
        bz_min = r - (z_c - z_min)
        bz_max = r + (z_max - z_c)

        by_min = r - (y_c - y_min)
        by_max = r + (y_max - y_c)

        bx_min = r - (x_c - x_min)
        bx_max = r + (x_max - x_c)

        # Extract the correctly clipped ball slice
        ball_slice = selected_ball[bz_min:bz_max, by_min:by_max, bx_min:bx_max]

        # Apply
        dilated_volume[z_min:z_max, y_min:y_max, x_min:x_max][ball_slice] = 1

    return dilated_volume


def generate_3d_synthetic_volume(image_path: str, uniform_dilation: bool = False, sphere_radius: int = 7,
                                 mode: int = 1, tilt: bool = False, image_binarize_threshold: int = 127):
    """
    Turn 2D image into a volumetric 3D NumPy array representation

    @:param image_path - a string of the path for the input image
    @:param uniform_dilation - whether to dilate uniformly or with varying radius
    @:param sphere_radius - radius of dilation sphere for volume generation
    @:param mode - indicates which skeletonization approach to use
                   1 := Thinning by Zhang (2D algorithm)
                   2 := Medial Axis
                   3 (or any other number) := Thinning
    @:param tilt - whether to tilt the embedded skeleton by 90°
    @:param image_binarize_threshold - which value to use as a barrier between 0 and 1

    @:returns skeleton_3d_volume - dilated 3D skeleton; the ready-generated synthetic data
    """

    # Load and convert image to grayscale
    image = Image.open(image_path).convert('L')

    # Thresholding to Binary NumPy Array
    # np.array(image) is natively (Rows, Cols) -> (Y, X) => add .T to transpose it to (Cols, Rows) -> (X, Y)
    # Assumes the object is black on a white background
    binary_2d = (np.array(image) < image_binarize_threshold).T

    # 2D Skeletonization
    if mode == 1:
        skeleton_2d = skeletonize(binary_2d, method='zhang')
    elif mode == 2:
        skeleton_2d, _ = medial_axis(binary_2d, return_distance=True)
    else:
        skeleton_2d = thin(binary_2d)
    # vis_result(skeleton_2d.T)  # Show 2D skeletonization result (uses again Image so .T needed)

    # Embed into 3D Space
    x, y = skeleton_2d.shape

    # The z-dimension needs to be large enough to hold the dilated spheres
    # Non-uniform dilation will offer balls of sized between [radius-2:radius+2]
    if uniform_dilation:
        z = 2 * sphere_radius + EMPTY_SLICES
    else:
        z = 2 * (sphere_radius + 4) + EMPTY_SLICES
    skeleton_3d = np.zeros((x, y, z), dtype=bool)

    # Place the 2D skeleton directly in the middle slice of the Z-axis (using integer division)
    skeleton_3d[:, :, z//2] = skeleton_2d

    # Tilt embedded skeleton - for now only by 90°
    if tilt:
        # show_all_slices(skeleton_3d)
        skeleton_3d = rotate(skeleton_3d, angle=90, axes=(0, 1), reshape=True)

    # Dilate 3D Skeleton to a Volume
    # skimage's ball() function to generate a 3D spherical structuring element
    if uniform_dilation:
        dilated_3d_volume = binary_dilation(skeleton_3d, structure=ball(sphere_radius))
    else:
        dilated_3d_volume = non_uniform_dilation(skeleton_mask=skeleton_3d,
                                                 radii_choices=list(range(sphere_radius, sphere_radius + 5)))

    # view_in_napari(colors=['white'], point_sizes=None, image={"Dilated volume": dilated_3d_volume})
    return dilated_3d_volume


