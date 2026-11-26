import sys
import numpy as np

"""
Class to store ordered voxels along traced subfilaments
And to allow the computation and storage of geometrical properties per subfilament
"""

class VoxelPath:
    def __init__(self, initial_capacity=128):
        # Currently use int16, but ToDo: make it dynamic
        self.coords = np.empty((initial_capacity, 3), dtype=np.int16)
        self.size = 0
        self.capacity = initial_capacity
        self.direction = None

    def add(self, voxel):
        if self.size == self.capacity:
            self.capacity *= 2
            new_coords = np.empty((self.capacity, 3), dtype=np.int16)
            new_coords[:self.size] = self.coords
            self.coords = new_coords

        self.coords[self.size] = voxel
        self.size += 1

    def add_whole_path(self, path: np.ndarray):
        # Check if provided path is empty
        if path.size == 0:
            return

        path_length = path.shape[0]

        # Check if we need to expand the array
        if self.capacity < self.size + path_length:

            # Multiply capacity by 2 until it is large enough
            while self.capacity < self.size + path_length:
                self.capacity *= 2

            # Reallocate and copy old data
            new_coords = np.empty((self.capacity, 3), dtype=np.int16)
            new_coords[:self.size] = self.coords
            self.coords = new_coords

        # Insert the new path and update size
        self.coords[self.size: self.size + path_length] = path
        self.size += path_length

    @property
    def data(self):
        """ Returns only taken-up part of array """
        return self.coords[:self.size]

    def length(self):
        """ Calculates total path length """
        return np.sum(np.linalg.norm(np.diff(self.data, axis=0), axis=1))

    def dir(self):
        if self.direction is None:
            raise Exception("No direction vector has been generated.")
        return self.direction

    """
    def curvature(self):
        # Estimates curvature using the discrete second derivative.
        # Returns an array of curvature magnitudes at each internal voxel.
        
        if self.size < 3:
            return np.zeros(0)

            # First derivative (tangent vectors)
        d1 = np.diff(self.data, axis=0)

        # Second derivative (rate of change of the tangents)
        d2 = np.diff(d1, axis=0)

        # Curvature magnitude at each point
        return np.linalg.norm(d2, axis=1)s
    """

    def set_avg_direction(self, start_pos: tuple, lower_limit: int, upper_limit: int):
        # Edge case: Filament is just 1 voxel (no direction)
        if self.size < 2:
            return np.array([0.0, 0.0, 0.0])

        if self.size < lower_limit:
            # if the whole filament is shorter than lower_limit, ignore it
            # print("Too small, None")
            return None

        # Get voxels from index lower_limit up to upper_limit (inclusive)
        if np.linalg.norm(np.array(start_pos) - self.data[0]) > \
                np.linalg.norm(np.array(start_pos) - self.data[-1]):
            # Reverse the coordinates
            target_voxels = self.data[::-1][lower_limit:(upper_limit + 1)]
        else:
            target_voxels = self.data[lower_limit:(upper_limit + 1)]

        # Calculate all vectors instantly
        vectors = target_voxels - start_pos

        # Average them along the columns (axis=0)
        avg_vector = np.mean(vectors, axis=0)

        # Normalize
        magnitude = np.linalg.norm(avg_vector)
        if magnitude == 0:
            self.direction = avg_vector
            return avg_vector  # Avoid dividing by 0

        self.direction = avg_vector/magnitude
        return avg_vector/magnitude
