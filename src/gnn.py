import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

def global_nearest_neighbor(data1: list, data2: list, similarity_fun: callable, min_similarity: float):
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
    # augment cost to add option for no associations
    hungarian_cost = np.ones((2*len1, 2*len2))
    M = 1e9

    for i in range(len1):
        for j in range(len2):
            similarity = similarity_fun(data1[i], data2[j])
            
            score = -similarity # Hungarian is trying to associate low similarity values, so negate
            if min_similarity is not None and similarity < min_similarity:
                score = M
            hungarian_cost[i,j] = score

    row_ind, col_ind = linear_sum_assignment(hungarian_cost)

    assignment = defaultdict(lambda: None)
    for idx1, idx2 in zip(row_ind, col_ind):
        if idx1 < len1 and idx2 < len2:
            assignment[idx2] = idx1

    return assignment