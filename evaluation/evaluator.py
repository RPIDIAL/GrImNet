import os, sys
import numpy as np
from evaluation.metrics import distance_error_from_distance_map, get_landmark_coordinates, distance_error_from_coordinate
sys.path.append('{}/../data_preparation'.format(os.path.dirname(os.path.realpath(__file__))))
from mesh_utility import get_mandible_landmark_list, get_midface_landmark_list, get_lmk_id_by_name, generate_roi_from_points, create_point_actor, create_line_actor, find_closest_point, generate_distance_map_actor_to_point
import pandas as pd

class Evaluator():
    def __init__(self, input_path, target_path, roi='mandible'):
        self.input_path = input_path
        self.target_path = target_path
        self.roi = roi
        if self.roi == 'mandible':
            self.landmark_list = get_mandible_landmark_list()
        else:
            self.landmark_list = get_midface_landmark_list()
        self.output_filename = "{0:s}/metrics.txt".format(self.input_path)

    def scan_input_path(self):
        caselist = {}
        for sub_fold in os.listdir(self.input_path):
            if not sub_fold.startswith('results_cv_'):
                continue
            for casename in os.listdir('{0:s}/{1:s}'.format(self.input_path, sub_fold)):
                if not casename.endswith('.npy'):
                    continue
                casename = casename[:-4]
                input_filename = '{0:s}/{1:s}/{2:s}.npy'.format(self.input_path, sub_fold, casename)
                target_filename = '{0:s}/{1:s}/img-aug-0/mesh_label.npy'.format(self.target_path, casename)
                mesh_filename = '{0:s}/{1:s}/img-aug-0/mesh.npz'.format(self.target_path, casename)
                if not os.path.exists(target_filename) or not os.path.exists(mesh_filename):
                    continue
                caselist[casename] = [input_filename, target_filename, mesh_filename]
        return caselist

    def evaluate_case(self, casename, input_filename, target_filename, mesh_filename):
        input_array = np.load(input_filename)

        target_array = np.load(target_filename)
        mesh_file = np.load(mesh_filename)
        points_coordinates = mesh_file['points']
        input_coordinates, _ = get_landmark_coordinates(input_array, points_coordinates, find_max=True)
        target_coordinates, valid_flag = get_landmark_coordinates(target_array, points_coordinates, find_max=True)
        dist_error = distance_error_from_coordinate(input_coordinates, target_coordinates, valid_flag)

        txt_filename = input_filename[0:-3] + 'txt'
        with open(txt_filename, 'w') as txt_file:
            for lmk_id in range(input_coordinates.shape[0]):
                if valid_flag[lmk_id]:
                    txt_file.write('{0:d} {1:.6f} {2:.6f} {3:.6f} {4:.6f}\n'.format(lmk_id, input_coordinates[lmk_id, 0], input_coordinates[lmk_id, 1], input_coordinates[lmk_id, 2], 1.0))

        return dist_error

    def evaluate(self):
        self.caselist = self.scan_input_path()
        self.df = None
        for casename in self.caselist:
            [input_filename, target_filename, mesh_filename] = self.caselist[casename]
            dist_error = self.evaluate_case(casename, input_filename, target_filename, mesh_filename)
            case_df = pd.DataFrame(self.landmark_list, columns=['Landmarks'])
            case_df['Dist. Error'] = dist_error
            case_df['Case'] = casename
            if self.df is None:
                self.df = case_df
            else:
                self.df = pd.concat([self.df, case_df], ignore_index=True)

        self.print_results()

    def print_results(self):
        lines = "Global results:\n"
        lines += " --- Distance error: {0:5.2f} ({1:5.2f})mm\n".format(self.df['Dist. Error'].mean(), self.df['Dist. Error'].std())
        lines += "\n"
        lines += "Landmark-wise results ({0:d} landmarks):\n".format(len(self.landmark_list))
        for lmk_name in self.landmark_list:
            lmk_df = self.df.loc[self.df['Landmarks'] == lmk_name]
            lines += " --- {0:12s}: {1:5.2f} ({2:5.2f})mm\n".format(lmk_name, lmk_df['Dist. Error'].mean(), lmk_df['Dist. Error'].std())
        lines += "\n"
        lines += "Case-wise results ({0:d} cases):\n".format(len(self.caselist))
        for casename in self.caselist:
            case_df = self.df.loc[self.df['Case'] == casename]
            lines += " --- {0:12s}: {1:5.2f} ({2:5.2f})mm\n".format(casename, case_df['Dist. Error'].mean(), case_df['Dist. Error'].std())
        print(lines)
        with open(self.output_filename, 'w') as output_file:
            output_file.write(lines)