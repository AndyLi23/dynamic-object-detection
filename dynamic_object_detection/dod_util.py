import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2 as cv
import os
import torch

def global_nearest_neighbor(data1: list, data2: list, cost_fn: callable, max_cost: float=None):
    """
    Associates data1 with data2 using the global nearest neighbor algorithm.

    Args:
        data1 (list): List of first data items
        data2 (list): List of second data items
        cost_fn (callable(item1, item2)): Function to compute the cost of associating two objects
        max_cost (float): Maximum cost to consider association

    Returns:
        a dictionary d such that d[i] = j means that data2[i] is associated with data1[j]
    """
    len1 = len(data1)
    len2 = len(data2)
    # Augment cost to add option for no associations
    hungarian_cost = np.ones((2*len1, 2*len2))
    M = 1e9

    for i in range(len1):
        for j in range(len2):
            score = cost_fn(data1[i], data2[j]) # Hungarian is trying to associate low scores, no negation needed
            
            if max_cost is not None and score > max_cost:
                score = M
            hungarian_cost[i,j] = score

    row_ind, col_ind = linear_sum_assignment(hungarian_cost)

    assignment = {}

    for idx1, idx2 in zip(row_ind, col_ind):
        if idx1 < len1 and idx2 < len2:
            assignment[idx2] = idx1

    return assignment

def global_nearest_neighbor_dynamic_objects(tracked_objects: dict, new_objects: list, cost_fn: callable, max_cost: float=None):
    """
    Associates tracked objects with new objects using the global nearest neighbor algorithm.

    Args:
        tracked_objects (dict): Dictionary of tracked objects
        new_objects (list): List of new objects
        cost_fn (callable): Function to compute the cost of associating two objects
        max_cost (float): Maximum cost to consider association

    Returns:
        a dictionary d such that d[i] = j means that new_object with id i is associated with tracked_object with id j
    """
    tracked_objects_list = list(tracked_objects.values())
    assignment = global_nearest_neighbor(tracked_objects_list, new_objects, cost_fn, max_cost)
    id_assignment = {new_objects[i].id : tracked_objects_list[j].id for i, j in assignment.items()}
    return id_assignment



def copy_params_file(parent_dir, params, args):
    params_copy_path = os.path.join(parent_dir, f'{os.path.basename(params.output)}.yaml')
    with open(args.params, 'r') as src_file, open(params_copy_path, 'w') as dest_file:
        dest_file.write(src_file.read())
    print(f'saved params file to {params_copy_path}')

def preprocess_depth(depth, depth_params):
    depth = depth.astype(np.float32)

    if depth_params.bilateral_smooth_depth is not None:
        d, sigmaColor, sigmaSpace = depth_params.bilateral_smooth_depth
        depth = cv.bilateralFilter(depth, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

    depth = depth / depth_params.depth_scale

    if depth_params.max_depth is not None: depth[depth > depth_params.max_depth] = 0
    return depth

def compute_relative_poses(poses):
    pose0 = poses[:-1]                                                                                  # (N, 4, 4)
    pose1_inv = torch.stack([torch.linalg.inv(pose) for pose in poses[1:]], dim=0)                      # (N, 4, 4)

    # T_1_0 = T_w_1^-1 @ T_w_0: transform from frame 0 to frame 1
    T_1_0 = pose1_inv @ pose0                                                                           # (N, 4, 4)      

    return T_1_0