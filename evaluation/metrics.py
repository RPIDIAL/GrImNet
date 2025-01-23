import numpy as np
import math

def get_landmark_coordinates(prob_array, point_coordinates, find_max=True):
    if find_max:
        max_prob_id = np.argmax(prob_array, axis=0)
        landmark_coordinates = point_coordinates[max_prob_id, :]
    else:
        tmp = np.percentile(prob_array, q=99.9, axis=0)
        tmp_mask = np.zeros_like(prob_array, dtype=np.uint8)
        landmark_coordinates = np.zeros((prob_array.shape[1], 3), dtype=point_coordinates.dtype)
        for i in range(len(tmp)):
            tmp_mask[:,i][prob_array[:,i] >= tmp[i]] = 1
            tmp_ind = np.nonzero(tmp_mask[:,i])
            tmp_coord = point_coordinates[tmp_ind[0], :]
            landmark_coordinates[i, :] = tmp_coord.mean(axis=0)
    max_prob = np.max(prob_array, axis=0)
    valid_flag = max_prob >= 1e-6
    return landmark_coordinates, valid_flag

def distance_error_from_coordinate(input_coordinates, target_coordinates, valid_flag=None):
    dist_error = np.sqrt(np.sum((input_coordinates - target_coordinates)**2, axis=1))
    if valid_flag is not None:
        for lmk_id, valid in enumerate(valid_flag):
            if not valid:
                dist_error[lmk_id] = math.nan

    return dist_error

def distance_error_from_distance_map(input_array, target_array, point_coordinates):
    input_coordinates = get_landmark_coordinates(input_array, point_coordinates, find_max=True)
    target_coordinates = get_landmark_coordinates(target_array, point_coordinates, find_max=True)
    dist_error = distance_error_from_coordinate(input_coordinates, target_coordinates)
    return dist_error
