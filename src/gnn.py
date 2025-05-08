import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

def global_nearest_neighbor(data1: list, data2: list, cost_fn: callable, max_cost: float=None):
    """
    Associates data1 with data2 using the global nearest neighbor algorithm.

    Args:
        data1 (list): List of first data items
        data2 (list): List of second data items
        similarity_fun (callable(item1, item2)): Evaluates the similarity between two items
        min_similarity (float): Minimum similarity required to associate two items

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

    assignment = defaultdict(lambda: None)
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
        cost_fn (callable): Function to compute the cost between two objects
        max_cost (float): Maximum cost for association

    Returns:
        a dictionary d such that d[i] = j means that new_objects[i] is associated with tracked_objects[j]
    """
    tracked_objects_list = list(tracked_objects.values())
    assignment = global_nearest_neighbor(tracked_objects_list, new_objects, cost_fn, max_cost)
    id_assignment = {new_objects[i].id : tracked_objects[j].id for i, j in assignment.items()}
    return id_assignment