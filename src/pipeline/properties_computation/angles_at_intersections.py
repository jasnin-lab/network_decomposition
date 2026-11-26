"""
The goal of this is to iterate the already done graph representation,
    compute angles between filaments in each intersection and
    finally based on this, to filter potential Arp2/3 intersections

The idea is to:
    - iterate each filament, connected to by the intersection
    - for each filament
        -> get its corresponding VoxelPath object
        -> build the direction vector:
            --> if needed, switch/reverse the order of voxels within the filament
            --> all 5 up to x with 20 < x < 40
            --> average all to obtain one
    - compute angle between each pair of neighboring filaments...
    - assign an attribute - dictionary (fil_num, fil_num) to angle
This will enable to filter intersections based on angles and connectivity
    --> then 1 function call would be enough to get all potential Arp2/3 intersection with angles ~ 70/110/180
"""

import sys, time, itertools, nrrd
import networkx as nx
import numpy as np


def compute_dirs_and_angles(graph: nx.Graph, lower_limit: int, upper_limit: int):
    """
    Iterates all intersection nodes in a graph and computes for each the angles between

    :param graph: graph representation of semantically segmented cytoskeletal network
    :param lower_limit: Lower limit of "straight line" after intersection
    :param upper_limit: Upper limit of "straight line" after intersection
    :return:
    """
    all_fil_paths = graph.graph['filament_paths']  # dict() with fil_num 2 VoxelPath object

    for node, attr in graph.nodes(data=True):
        if attr.get('node_type') == 'intersection':  # filter for intersection nodes
            for fil in attr.get('connections').keys():

                fil_path = all_fil_paths[fil]

                # Calculate averaged and normalized direction vector and store it in each VoxelPath object
                avg_norm_direction = fil_path.set_avg_direction(start_pos=attr.get('coords'), lower_limit=lower_limit,
                                                                upper_limit=upper_limit)

                # Updated filaments to be in a dictionary, and we need to assign now the vector to the filament number
                attr.get('connections').update({fil: avg_norm_direction})

            # Once averaged direction unit vectors have been computed
            # Calculate angle between filament vectors and store them...
            """ For now we store the angles as another attribute at intersection nodes """
            # Assign a dict - the angles to intersections as an attribute
            graph.nodes[node]['angles'] = get_all_pairwise_angles(attr.get('connections'))



def get_all_pairwise_angles(fil_num_2_direction: dict):
    """
    Calculate all 3D angles between all filaments for an intersection
    :return: a dictionary (v1, v2) : angle
    """

    if len(fil_num_2_direction) < 2:
        return {}  # We need at least 2 filaments to calculate an angle

    results = {}

    for fil_1, fil_2 in itertools.combinations(fil_num_2_direction.keys(), 2):
        if fil_num_2_direction[fil_1] is None or fil_num_2_direction[fil_2] is None:
            continue

        dot_prod = np.dot(fil_num_2_direction[fil_1], fil_num_2_direction[fil_2])
        dot_prod = np.clip(dot_prod, -1.0, 1.0)  # Forces the value to stay structly between those boundaries
        # to avoid 1.0000000000000000002

        angle = np.degrees(np.arccos(dot_prod))

        results[(fil_1, fil_2)] = angle

    return results


def get_intersection_candidates(graph: nx.Graph, target_angles: tuple, threshold: int = 10):
    candidates = []

    for node, attr in graph.nodes(data=True):
        sorted_targets = sorted(target_angles)

        if attr.get('node_type') == 'intersection':
            connections = attr.get('connections')

            if len(connections) == len(target_angles):
                angles_dict = graph.nodes[node]['angles']

                if len(angles_dict.keys()) == (len(target_angles)*(len(target_angles) - 1))/2:  # ToDo: extend by the expected number of angles based on angles to check
                    sorted_angles = sorted(angles_dict.values())

                    # Check if EVERY angle is within the tolerance
                    is_match = all(
                        abs(angle - target) <= threshold
                        for angle, target in zip(sorted_angles, sorted_targets))

                    if is_match:
                        # Append the node ID, attributes can be accessed
                        candidates.append(node)

    return candidates
