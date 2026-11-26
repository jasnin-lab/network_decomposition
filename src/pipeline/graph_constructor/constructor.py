"""
Construct a graph (NetworkX) of the skeleton of the segmented volume
Iterate through the whole skeleton and simultaneously define a homeomorphic graph,
while storing geometrical information (voxel coordinates)
"""
import sys, time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.pipeline.utils_vis import *

 
def add_edge_adv(graph: nx.Graph, node_1, node_2, intermediate):
    if graph.has_edge(node_1, node_2):
        graph.add_node(f"i_{intermediate}",
                       node_type='intermediate')
        graph.add_edge(node_1, f"i_{intermediate}")
        graph.add_edge(f"i_{intermediate}", node_2)
        intermediate += 1
    else:
        graph.add_edge(node_1, node_2)

    return intermediate



def graph_construction(skeleton: np.ndarray, start_coordinates: np.ndarray,
                       tp_intersections, int_nc, mask, fp_2_true_intersection, voxel_size=None):

    # Initialize identically sized numpy array to label voxels already visited and perform look-ups
    voxel_lookup = np.zeros_like(skeleton)

    # Initialize empty graph and iteratively add nodes and edges
    graph = nx.Graph()

    # Store the shape of the mask as a graph attribute (== the shape of the skeleton)
    graph.graph["mask_shape"] = skeleton.shape

    # Initialize dictionary to assign fil_num:VoxelPath
    fil_id_2_path = dict()

    # Running index for nodes and filaments
    node_id = 1
    intermediate_id = 1
    fil_num = 1

    """ Iterate intersection voxels 
    Initialize for each a node and assign each one:
                - id
                - type of node ("intersection")
                - voxel coordinates
                - list of ids of filaments connected to it 
    """

    intersection_lookup = dict()
    for intersection_coords in tp_intersections:
        graph.add_node(node_id,
                       node_type='intersection',
                       connections=dict(),
                       coords=tuple(int(x) for x in intersection_coords))
        intersection_lookup.update({tuple(intersection_coords): node_id})

        # Prepare for next iteration
        node_id += 1

    """
    If this list is empty we get a shape error later. To avoid it the simplest way:
    add an element, which is assumed to not be close an intersection
    """
    if len(tp_intersections) == 0:
        tp_intersections.append([0, 0, 0])

    """ Start Skeleton Iteration:
        --> We expect to have already visited start voxels while iterating 
             when having start-start filaments -> the end of a filament has been also stores as a start point
             but it would have been already iterated and labeled as visited 
                => check at each iteration if start has been visited
        --> Otherwise, get a start positions - label it as visited and trace its filament until end/intersection
            - at each position/voxel - jump to next voxel and return one based on its type
    """
    for vox_coords in start_coordinates:
        if voxel_lookup[tuple(vox_coords)] == 1:
            continue  # Start voxel already visited

        # Start voxel not yet visited => should be labeled as visited and its filament should be traced
        voxel_lookup[tuple(vox_coords)] = 1

        # Store for filament path
        path = voxel_path.VoxelPath(initial_capacity=20)
        path.add(tuple(vox_coords))

        # Add start point node to graph and filament number as its attribute;
        # Later connect it to corresponding intersection or end node
        graph.add_node(node_id,
                       node_type='start',
                       fil_num=fil_num,
                       coords=tuple(int(x) for x in vox_coords))
        node_id += 1

        # Iterate filament and store coordinates until end or intersection
        while True:
            next_vox = assign_next_voxel(skeleton, tuple(vox_coords), voxel_lookup,
                                         tp_intersections, fp_2_true_intersection)  # next_vox is tuple()

            if next_vox is None:  # End of filament without reaching an intersection
                # Initialize a final node to graph and connect the start one to it
                graph.add_node(node_id,
                               node_type='end',
                               fil_num=fil_num,
                               coords=tuple(int(x) for x in vox_coords))
                intermediate_id = add_edge_adv(graph, node_id, node_id-1, intermediate_id)

                # Store path in dictionary
                fil_id_2_path.update({fil_num: path})

                node_id += 1
                fil_num += 1  # Increment id for the new filament - this one ended
                break
            elif np.any(np.all(tp_intersections == np.array(next_vox), axis=1)):
                """ 
                Next (deterministic) voxel is a TP intersection
                    => Add edge to corresponding intersection node and end filament tracing 
                """
                intermediate_id = add_edge_adv(graph, (node_id-1), intersection_lookup[tuple(next_vox)], intermediate_id)

                graph.nodes[intersection_lookup[tuple(next_vox)]]["connections"].update({fil_num: None})

                # Store path in dictionary
                fil_id_2_path.update({fil_num: path})

                fil_num += 1
                break

            """
            The above if-statements do not handle the case of the next voxel being a FP intersection voxel...
            ToDo: maybe add it as it is a case at which we should end the current tracing
            - one negative of the current state of the pipeline/logic -> the skipped 2nd layer FP intersections
            """
            # Jump to new voxel that is not an intersection (BUT could be a FP intersection)
            voxel_lookup[next_vox] = 1
            # Store for filament path
            path.add(tuple(next_vox))
            # Change current voxel to the new one
            vox_coords = next_vox

        # Here = results of a break-statement => end/intersection reached => so we move on to next start pos/filament

    # View in napari which voxels have been visited
    """
    print("View which voxels have been iterated (in pink) from the start positions")
    view_in_napari(colors=['blue', 'white', 'pink', None, 'white'], point_sizes=[1],
                   image={"Segmentation": mask, "Skeleton": skeleton, "Visited voxels": voxel_lookup},
                   labels={"Potential Intersections": int_nc},
                   points={"Mean-Shifted Intersections": tp_intersections})
    """

    prev_dif = 0
    while skeleton.sum() != voxel_lookup.sum():
        # A break-statement to not fall into an endless loop
        cur_dif = skeleton.sum() - voxel_lookup.sum()

        if prev_dif == cur_dif:
            print("No changes made, No new filaments")
            break
        # Update reference point
        prev_dif = cur_dif

        """ Use intersections as start positions
        However, an intersection may be able to go in multiple directions
        => an intersection should be labeled only after all possible direction taken 
        """

        # Iterate each intersection voxel as a start position
        for inters_start in tp_intersections:
            v1 = tuple(inters_start)  # (intersection) start voxel coordinates  [V1]
            if voxel_lookup[v1] == 1:  # Check if intersection already labeled as checked
                continue

            v2 = assign_next_voxel_ints(skeleton, v1, voxel_lookup)  # [V2]
            if v2 is None:  # No where new to go from the intersection, Intersection already labeled as visited
                continue

            # V2 is not None => there is a direction to trace
            voxel_lookup[tuple(v2)] = 1  # Label v2 visited

            # Find v3 and if multiple voxel candidates -> dot product to check angle to previous voxels and choose
            v3 = assign_next_voxel_test(skeleton, v2, voxel_lookup, v1)
            if np.any(np.all(tp_intersections == np.array(v3), axis=1)) and v3 == v1:
                # If v3 goes to same intersection as the start and this being the "best" and only option
                continue
            elif np.any(np.all(tp_intersections == np.array(v3), axis=1)) and v3 != v1:
                # Special case --> 1-voxel filament (v2) between different intersections; V3 a different TP intersection
                graph.nodes[intersection_lookup[v1]]["connections"].update({fil_num: None})
                graph.nodes[intersection_lookup[v3]]["connections"].update({fil_num: None})

                intermediate_id = add_edge_adv(graph, intersection_lookup[v1], intersection_lookup[v3], intermediate_id)

                path = voxel_path.VoxelPath(initial_capacity=20)
                path.add(v2)
                path.add(v3)
                fil_id_2_path.update({fil_num: path})
                fil_num += 1
                continue
            elif v3 in fp_2_true_intersection.keys():
                # Special case in which we stay within the FP intersection cloud around a TP intersection
                if fp_2_true_intersection[v3] == v1:
                    continue

            # A quick test if v3 is None - stop to detect issue or special case
            if v3 is None:
                raise Exception(f"[V3] is None with V1 {tuple(v1)} and V2 {tuple(v2)}")

            voxel_lookup[tuple(v3)] = 1
            graph.nodes[intersection_lookup[v1]]["connections"].update({fil_num: None})  # => there is sth to trace

            # Store for filament path
            path = voxel_path.VoxelPath(initial_capacity=20)
            path.add(v2)
            path.add(v3)

            while True:  # Trace filament/line until FP/TP intersection
                next_vox = assign_next_voxel(skeleton, v3, voxel_lookup,
                                             tp_intersections, fp_2_true_intersection)

                if next_vox is None:  # End condition
                    """ 
                    No unvisited voxels to trace => the inters-inters line has reached already iterated voxels
                    Check whether there is a checked (FP) intersection voxel.
                    If so - assign the line to it and end iteration
                    """

                    intersection_voxel_neighbour = check_direct_neighbours(skeleton, v3,
                                                                           tp_intersections, fp_2_true_intersection)
                    intermediate_id = add_edge_adv(graph, intersection_lookup[v1],
                                                   intersection_lookup[tuple(intersection_voxel_neighbour)], intermediate_id)

                    graph.nodes[intersection_lookup[tuple(intersection_voxel_neighbour)]]["connections"].update({fil_num: None})

                    # Store path in dictionary
                    fil_id_2_path.update({fil_num: path})

                    fil_num += 1  # Increment to store new id for the new filament. This one ended => move to a new one
                    break
                elif np.any(np.all(tp_intersections == np.array(next_vox), axis=1)):  # next voxel is TP intersection
                    # Add edge to corresponding intersection node and end filament tracing
                    intermediate_id = add_edge_adv(graph, intersection_lookup[v1],
                                                   intersection_lookup[tuple(next_vox)], intermediate_id)

                    graph.nodes[intersection_lookup[tuple(next_vox)]]["connections"].update({fil_num: None})

                    # Store path in dictionary
                    fil_id_2_path.update({fil_num: path})

                    fil_num += 1
                    break

                # Jump to new voxel
                voxel_lookup[next_vox] = 1

                # Store for filament path
                path.add(next_vox)

                # Change current voxel to the new one
                v3 = next_vox

            # Once iterated the whole line, we can move on to next

    # Otherwise all start points and intersections are iterated, check again:
    if not skeleton.sum() == voxel_lookup.sum():
        differences = np.argwhere((skeleton == 1) & (voxel_lookup == 0))

        # Check if all coordinates do not have any neighbors
        isolated = 0
        not_isolated = 0
        for x, y, z in differences:
            neighborhood = skeleton[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]

            # True if center is 1 and all 26 neighbors are 0
            if neighborhood.sum() == 1:
                voxel_lookup[(x, y, z)] = 1
                isolated += 1
            else:
                not_isolated += 1

        # Visually examine
        """
        view_in_napari(colors=['blue', 'white', None, 'red', 'white'], point_sizes=[1, 1],
                       image={"Segmentation": mask, "Skeleton": skeleton},
                       labels={"Potential Intersections": int_nc},
                       points={"Differences": differences, "Mean-Shifted Intersections": tp_intersections})
        """

    """
    view_in_napari(colors=['blue', 'white', None, 'red', 'green'], point_sizes=[3, 1],
                   image={"Segmentation": mask, "Skeleton": skeleton},
                   labels={"Potential Intersections": int_nc},
                   points={"Mean-Shifted Intersections": tp_intersections,
                           "Start position": start_coordinates})
    """

    while not skeleton.sum() == voxel_lookup.sum():
        differences = np.argwhere((skeleton == 1) & (voxel_lookup == 0))
        # Take a random unvisited voxel and check if it is isolated
        current_voxel = differences[0]

        # Store for filament path
        path = voxel_path.VoxelPath(initial_capacity=20)
        path.add(tuple(current_voxel))

        # Will be traced => label as visited from beginning
        voxel_lookup[tuple(current_voxel)] = 1

        # Trace its filament/line and connect to intersections
        # Start from one side and iterate until next voxel is a FP intersection
        directions = get_direction_coords(skeleton, tuple(current_voxel), voxel_lookup)

        if len(directions) == 1:
            # Get neighboring FP to TP identity
            intersection_voxel_neighbour_1 = check_direct_neighbours(skeleton, tuple(current_voxel), tp_intersections, fp_2_true_intersection)

            # Connect filament to it
            graph.nodes[intersection_lookup[tuple(intersection_voxel_neighbour_1)]]["connections"].update({fil_num: None})

            # Start tracing to the only possible direction until end is reached
            while True:
                next_vox = assign_next_voxel(skeleton, tuple(current_voxel), voxel_lookup,
                                             tp_intersections, fp_2_true_intersection)  # next_vox is tuple()

                if next_vox is None:
                    # No other unvisited next voxels => search again for the nearest FP to assign to true intersection
                    intersection_voxel_neighbour_2 = check_direct_neighbours(skeleton, tuple(current_voxel),
                                                                             tp_intersections, fp_2_true_intersection)
                    intermediate_id = add_edge_adv(graph, intersection_lookup[tuple(intersection_voxel_neighbour_1)],
                                                   intersection_lookup[tuple(intersection_voxel_neighbour_2)], intermediate_id)

                    graph.nodes[intersection_lookup[tuple(intersection_voxel_neighbour_2)]]["connections"].update({fil_num: None})
                    break
                elif np.any(np.all(tp_intersections == np.array(next_vox), axis=1)):  # next voxel -> intersection voxel
                    intermediate_id = add_edge_adv(graph, intersection_lookup[tuple(intersection_voxel_neighbour_1)],
                                                   intersection_lookup[tuple(next_vox)], intermediate_id)

                    graph.nodes[intersection_lookup[tuple(next_vox)]]["connections"].update({fil_num: None})
                    voxel_lookup[tuple(next_vox)] = 1
                    break

                voxel_lookup[tuple(next_vox)] = 1
                path.add(tuple(next_vox))
                current_voxel = next_vox

            # Store path in dictionary
            fil_id_2_path.update({fil_num: path})
            fil_num += 1
        elif len(directions) == 2:
            # Trace direction 1 until FP intersection or end
            #    and connect current filament number to end of this direction
            path_dir_1 = voxel_path.VoxelPath(initial_capacity=20)
            path_dir_2 = voxel_path.VoxelPath(initial_capacity=20)
            current_voxel = tuple(directions[0])
            voxel_lookup[current_voxel] = 1

            circle = False
            while True:
                next_vox = assign_next_voxel(skeleton, tuple(current_voxel), voxel_lookup,
                                             tp_intersections, fp_2_true_intersection)  # next_vox is tuple()

                if next_vox is None:
                    # No other unvisited next voxels => search again for the nerest FP to assign to true intersection
                    intersection_voxel_neighbour_2 = check_direct_neighbours(skeleton, tuple(current_voxel), tp_intersections, fp_2_true_intersection)

                    if intersection_voxel_neighbour_2 is None:
                        circle = True
                        # Add circle to graph
                        graph.add_node(node_id, fil_num=fil_num, coords=tuple(directions[0]))
                        graph.add_edge(node_id, node_id)
                        node_id += 1

                        # Assign geometry information
                        fil_id_2_path.update({fil_num: path_dir_1})
                        fil_num += 1
                        break

                    graph.nodes[intersection_lookup[tuple(intersection_voxel_neighbour_2)]]["connections"].update({fil_num: None})
                    inters_one_side = intersection_lookup[tuple(intersection_voxel_neighbour_2)]
                    break
                elif np.any(np.all(tp_intersections == np.array(next_vox), axis=1)):  # next voxel is an assigned intersection voxel
                    graph.nodes[intersection_lookup[tuple(next_vox)]]["connections"].update({fil_num: None})
                    voxel_lookup[tuple(next_vox)] = 1
                    inters_one_side = intersection_lookup[tuple(next_vox)]
                    break

                voxel_lookup[tuple(next_vox)] = 1
                path_dir_1.add(tuple(next_vox))
                current_voxel = next_vox

            # Edge case of circles
            if circle:
                break

            # Trace direction 2 until FP intersection or end
            #   and connect current filament number to end of this direction
            current_voxel = tuple(directions[1])
            voxel_lookup[current_voxel] = 1

            while True:
                next_vox = assign_next_voxel(skeleton, tuple(current_voxel), voxel_lookup,
                                             tp_intersections, fp_2_true_intersection)  # next_vox is tuple()

                if next_vox is None:
                    # No other unvisited next voxels => search again for the nearest FP to assign to true intersection
                    intersection_voxel_neighbour_2 = check_direct_neighbours(skeleton, tuple(current_voxel), tp_intersections, fp_2_true_intersection)
                    graph.nodes[intersection_lookup[tuple(intersection_voxel_neighbour_2)]]["connections"].update({fil_num: None})
                    inters_other_side = intersection_lookup[tuple(intersection_voxel_neighbour_2)]
                    break
                elif np.any(np.all(tp_intersections == np.array(next_vox), axis=1)):  # next voxel is an assigned intersection voxel
                    graph.nodes[intersection_lookup[tuple(next_vox)]]["connections"].update({fil_num: None})
                    voxel_lookup[tuple(next_vox)] = 1
                    inters_other_side = intersection_lookup[tuple(next_vox)]
                    break

                voxel_lookup[tuple(next_vox)] = 1
                path_dir_2.add(tuple(next_vox))
                current_voxel = next_vox

            # once both have been connected, increment filament number and (MAKE sure to label all as visited)
            intermediate_id = add_edge_adv(graph, inters_one_side, inters_other_side, intermediate_id)

            # Store each path in a separate VoxelPath object and at the end combine both
            # Assess the non-continuous path tracing of the subfilament by
            path1 = path_dir_1.data()[::-1]
            path2 = path_dir_2.data()
            path.add_whole_path(path1)
            path.add_whole_path(path2)
            fil_id_2_path.update({fil_num: path})
            fil_num += 1
        elif len(directions) == 0:
            if tuple(current_voxel) in fp_2_true_intersection.keys():
                voxel_lookup[tuple(current_voxel)] = 1
            else:
                voxel_lookup[tuple(current_voxel)] = 1
        else:  # Error handling just in case
            raise Exception(f"ERROR: unexpected number of missed directions. Currently {len(directions)}")

    # Add graph attributes
    graph.graph["all_filaments"] = fil_num - 1
    graph.graph["filament_paths"] = fil_id_2_path
    graph.graph["voxel_size"] = voxel_size  # Assign voxel size coordinates to graph as an attribute

    return graph


def get_global_neighbor_coords(skeleton: np.ndarray, curr_voxel: tuple):
    """ ToDo: function description """
    if len(curr_voxel) != 3:
        raise ValueError(f"Expected curr_voxel to have 3 dimensions (z, y, x), but got {len(curr_voxel)}")

    # Edges of mask
    x_max, y_max, z_max = skeleton.shape
    x, y, z = curr_voxel

    # Define 3x3x3 boundaries with safe edge clipping so we don't look after the array edges
    x_min = max(0, x - 1)
    x_end = min(x_max, x + 2)

    y_min = max(0, y - 1)
    y_end = min(y_max, y + 2)

    z_min = max(0, z - 1)
    z_end = min(z_max, z + 2)

    # Extract the local 3x3x3 grid
    #   the coordinates in this one will not match those in the original skeleton mask :(
    #   maybe all they need to do is to define in which directions a step is possible
    local_grid = skeleton[x_min:x_end, y_min:y_end, z_min:z_end]

    # Find all local coordinates with True value => skeleton voxels
    local_coords = np.argwhere(local_grid)

    # Translate local coordinates back to global and return them
    return local_coords + np.array([x_min, y_min, z_min])


def assign_next_voxel_start_pos(skeleton: np.ndarray, curr_voxel: tuple, check_up: np.ndarray, intersection_map: dict):
    """ ToDo: function description """
    global_coords = get_global_neighbor_coords(skeleton, curr_voxel)

    # Iterate through the global coordinates
    candidates = []
    for candidate in global_coords:
        if check_up[tuple(candidate)] == 0:
            candidates.append(candidate)

    # Check collected candidates
    if len(candidates) == 0:
        return None
    elif len(candidates) == 1:
        # Need to check whether next voxel is an FP intersection voxel to return the corresponding True intersection
        # Otherwise -> whether True intersection or within-filament voxel -> they will be handled accordingly
        if tuple(candidates[0]) in intersection_map.keys():
            # Current deterministic next voxel is an FP intersection
            #   => label it as visited and jump to TP intersection
            check_up[tuple(candidates[0])] = 1
            return intersection_map[tuple(candidates[0])]  # should return coordinates as a tuple
        else:
            return tuple(candidates[0])
    else:
        raise Exception("Error in start position iteration - multiple possible next voxels")


def assign_next_voxel(skeleton: np.ndarray, curr_voxel: tuple, check_up: np.ndarray, intersections: np.ndarray, intersection_map: dict):
    """ToDo: add function description"""
    global_coords = get_global_neighbor_coords(skeleton, curr_voxel)

    # Iterate through the global coordinates
    candidates = []
    for candidate in global_coords:
        if check_up[tuple(candidate)] == 0:
            candidates.append(candidate)

    # Check gesammelte candidates
    if len(candidates) == 0:
        return None
    elif len(candidates) == 1:
        return tuple(candidates[0])
    else:

        if np.any(np.all(intersections == curr_voxel, axis=1)):
            return curr_voxel  # Return the current voxel, since it is already an intersection and end filament tracing
        else:  # If current voxel is not an intersection -> exit as handling with FP intersections not yet handled
            # Assign it its true intersection voxel and return it to end the tracing
            return intersection_map[curr_voxel]  # Return true intersection voxel corresponding to the FP intersection


def assign_next_voxel_ints(skeleton: np.ndarray, curr_voxel: tuple, check_up: np.ndarray):  # intersections: np.ndarray
    """V1 --> V2"""
    global_coords = get_global_neighbor_coords(skeleton, curr_voxel)

    # Iterate through the global coordinates  ToDo: should be able to automate this step (numpy...)
    candidates = []
    for candidate in global_coords:
        if check_up[tuple(candidate)] == 0 and (not (candidate == np.array(curr_voxel)).all()):
            candidates.append(candidate)
    if len(candidates) == 0:
        check_up[curr_voxel] = 1  # Label intersection voxel as visited
        return None
    else:
        return candidates[0]
        # No matter if one or multiple are possible, return the first possible; others will be iterated


def assign_next_voxel_test(skeleton: np.ndarray, curr_voxel: tuple, check_up: np.ndarray, intersection_vox_coords: tuple):  # intersections: np.ndarray
    """ V2 --> V3 """
    global_coords = get_global_neighbor_coords(skeleton, curr_voxel)

    # Iterate through the global coordinates  ToDo: should be able to automate this step (numpy...)
    candidates = []
    for candidate in global_coords:
        if check_up[tuple(candidate)] == 0:
            candidates.append(candidate)

    if len(candidates) == 0:
        return None
    elif len(candidates) == 1:
        return tuple(candidates[0])
    else:
        # Multiple possible directions (currently at a FP intersection)
        # => compute angle test to decide in which direction to continue

        # intersection_vox_coords [V1] and curr_voxel [V2] are known
        direction_vector = np.array(curr_voxel) - np.array(intersection_vox_coords)  # V2 - V1

        """ Iterate all possible V3 voxels and return the one with least angle to the original direction """
        direction = None  # Place holder for V3
        min_angle = 500.0  # a value higher than the highest possible angle
        for test in candidates:
            direction_test = test - np.array(curr_voxel)
            curr_angle = calculate_angle(direction_vector, direction_test)

            if curr_angle < min_angle:
                min_angle = curr_angle
                direction = test

        return tuple(direction)


def calculate_angle(vec1, vec2):
    # Convert to Unit Vectors
    unit_v1 = vec1 / np.linalg.norm(vec1)
    unit_v2 = vec2 / np.linalg.norm(vec2)

    # Calculate the Dot Product
    # np.clip to prevent tiny floating point errors like 1.00000002 which can crash arccos
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)

    # Calculate theta (in radians) using arccos
    angle_rad = np.arccos(dot_product)

    # Convert to degrees for easier debugging and reading
    return np.degrees(angle_rad)


def check_direct_neighbours(skeleton: np.ndarray, curr_voxel: tuple, intersections: np.ndarray, intersection_map: dict):
    global_coords = get_global_neighbor_coords(skeleton, curr_voxel)

    for potential in global_coords:  # check each skeleton voxel
        if np.any(np.all(intersections == potential, axis=1)):
            return tuple(potential)
        elif tuple(potential) in intersection_map:
            if np.any(np.all(intersections == intersection_map[tuple(potential)], axis=1)):  # neighbor FP intersection?
                return intersection_map[tuple(potential)]

    return None


def get_direction_coords(skeleton: np.ndarray, curr_voxel: tuple, check_up:np.ndarray):
    global_coords = get_global_neighbor_coords(skeleton, curr_voxel)

    candidates = list()
    for candidate in global_coords:
        if check_up[tuple(candidate)] == 0:
            candidates.append(candidate)

    return candidates
