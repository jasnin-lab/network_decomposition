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
        if len(self.data) < 2 or len(self.data) <= lower_limit:
            self.direction = None
            return None

        # Orient voxels starting from start_pos
        start_arr = np.asarray(start_pos)
        if np.linalg.norm(start_arr - self.data[0]) > np.linalg.norm(start_arr - self.data[-1]):
            target_voxels = self.data[::-1][lower_limit:(upper_limit + 1)]
        else:
            target_voxels = self.data[lower_limit:(upper_limit + 1)]

        # Safety guard (also catches upper_limit < lower_limit)
        if len(target_voxels) == 0:
            self.direction = None
            return None

        vectors = target_voxels - start_arr
        avg_vector = np.mean(vectors, axis=0)

        magnitude = np.linalg.norm(avg_vector)
        if magnitude == 0:
            self.direction = None
            return None

        self.direction = avg_vector / magnitude
        return self.direction
