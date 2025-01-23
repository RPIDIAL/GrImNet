import torch
from torch_geometric.data import Data, Dataset
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import sys, os
sys.path.append('{}/../data_preparation'.format(os.path.dirname(os.path.realpath(__file__))))
from mesh_utility import get_midface_landmark_list, get_mandible_landmark_list, get_landmark_coordinate, resample_image, generate_mesh_coordinate_grid
import pandas as pd
import SimpleITK as sitk
import math

class GraphDataset(Dataset):
    def __init__(self, case_list, enable_augmentation, subset_name):
        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None)
        self.enable_augmentation = enable_augmentation
        self.aug_num = 10
        self.data_list = []
        assert subset_name == 'midface' or subset_name == 'mandible', "The value of 'subset_name' can only be 'mandible' or 'midface'."
        if subset_name == 'midface':
            self.lmk_name_list = get_midface_landmark_list()
        else:
            self.lmk_name_list = get_mandible_landmark_list()
        for [casename, case_path] in tqdm (case_list, desc="Loading data ..."):
            graph_data = self.load_case_data(casename, case_path)
            self.data_list.append(graph_data)
            if self.enable_augmentation:
                for aug_id in range(self.aug_num):
                    graph_data = self.load_case_data(casename, case_path, aug_id+1)
                    self.data_list.append(graph_data)

    def len(self):
        return len(self.data_list)
    
    def load_case_data(self, casename, case_path, aug_id=0):
        mesh_fn = "{0:s}/img-aug-{1:d}/mesh.npz".format(case_path, aug_id)
        image_fn = "{0:s}/img-aug-{1:d}/image.npy".format(case_path, aug_id)
        resampled_image_fn = "{0:s}/img-aug-{1:d}/resampled_image.npy".format(case_path, aug_id)
        mask_fn = "{0:s}/img-aug-{1:d}/solid_label.nii.gz".format(case_path, aug_id)
        mesh_label_fn = "{0:s}/img-aug-{1:d}/mesh_label.npy".format(case_path, aug_id)
        lmk_df_fn = "{0:s}/img-aug-{1:d}/lmk.csv".format(case_path, aug_id)
        lmk_df = pd.read_csv(lmk_df_fn)

        mesh_file = np.load(mesh_fn)
        
        rel_features = torch.tensor(mesh_file['rel_features'], dtype=torch.float)
        rel_features = (rel_features - rel_features.mean(dim=0)) / rel_features.std(dim=0)
        normals = torch.tensor(mesh_file['normals'], dtype=torch.float)
        x = torch.cat([rel_features, normals], dim=1)
        edge_index = torch.tensor(mesh_file['edge_index'], dtype=torch.long)
        edge_weight = torch.tensor(mesh_file['edge_weight'], dtype=torch.long)
        mesh_label = np.load(mesh_label_fn)
        lmk_inds = torch.tensor(np.argmax(mesh_label, axis=0), dtype=torch.long)
        for lmk_id in range(mesh_label.shape[1]):
            if mesh_label[:,lmk_id].max() < 1e-6:
                lmk_inds[lmk_id] = -1

        y = torch.tensor(mesh_label, dtype=torch.float)
        batch = torch.zeros(x.shape[0]).long()
        
        enforce_resampling = False
        label_sitk_image = sitk.ReadImage(mask_fn)
        image_size = np.array(label_sitk_image.GetSize(), dtype=int)
        image_spacing = np.array(label_sitk_image.GetSpacing(), dtype=float)
        image_origin = np.array(label_sitk_image.GetOrigin(), dtype=float)
        new_size = np.array([96, 96, 96], dtype=image_size.dtype)
        new_spacing = np.array([2.5, 2.5, 2.5], dtype=image_spacing.dtype)
        new_origin = image_origin + 0.5 * image_size * image_spacing - 0.5 * new_size * new_spacing
        if not os.path.exists(resampled_image_fn) or enforce_resampling:
            image = sitk.GetImageFromArray(np.load(image_fn))
            image.SetSpacing(image_spacing)
            image.SetOrigin(image_origin)
        
            new_image = resample_image(image, new_size, new_spacing, new_origin, interpolation='linear')
            np.save(resampled_image_fn, sitk.GetArrayFromImage(new_image))

        image_start_pos = new_origin.copy()
        image_end_pos = image_start_pos + new_size * new_spacing
        
        points = mesh_file['points']
        grid_pos = 2.0 * (points - image_start_pos) / (image_end_pos - image_start_pos) - 1.0
        grid_pos = torch.tensor(grid_pos, dtype=torch.float).view(1, 1, 1, -1, 3)

        points = torch.tensor(points, dtype=torch.float)
        point_center = points.mean(axis=0)
        point_std = points.std()
        points = (points - point_center) / point_std

        image_data = np.load(resampled_image_fn)
        image_data = (image_data - image_data.mean()) / image_data.std()
        coord_grid = generate_mesh_coordinate_grid(new_origin, new_size, new_spacing)
        dist_map = np.zeros((len(self.lmk_name_list), image_data.shape[0], image_data.shape[1], image_data.shape[2]), dtype=float)
        image_data = image_data.reshape(1, 1, image_data.shape[0], image_data.shape[1], image_data.shape[2])
        image_data = torch.tensor(image_data, dtype=torch.float)
        for lmk_id, lmk_name in enumerate(self.lmk_name_list):
            target_lmk_coord = get_landmark_coordinate(lmk_df, lmk_name)
            if math.isnan(target_lmk_coord[0]) or math.isnan(target_lmk_coord[1]) or math.isnan(target_lmk_coord[2]):
                dist_map[lmk_id, :] = 1e8
            else:
                dist_map[lmk_id, :] = np.sqrt(np.sum((coord_grid - target_lmk_coord)**2, axis=3))
        dist_map = np.exp(-0.01*dist_map**2)
        dist_map = torch.tensor(dist_map, dtype=torch.float).unsqueeze(0)
        
        graph_data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y, batch=batch, casename=casename, lmk_inds=lmk_inds, points=points, image_data=image_data, dist_map=dist_map, grid_pos=grid_pos, image_spacing=new_spacing, image_origin=new_origin)

        return graph_data
    
    def get(self, idx):
        graph_data = self.data_list[idx].detach()

        return graph_data