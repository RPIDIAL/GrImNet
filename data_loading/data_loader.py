import torch
import os, sys 
sys.path.append('../')
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataListLoader
import random
from data_loading.hybrid_dataset import GraphDataset

class CrossValidationDataLoader():
    def __init__(self, data_source_path, fold_num, batch_size, num_workers, subset_name):
        
        self.data_source_path = data_source_path
        self.fold_num = fold_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subset_name = subset_name
        fold_file_name = self.load_or_create_fold_file()
        self.data_list = self.load_data_list(fold_file_name)

        folds_size = [len(x) for x in self.data_list.values()]
        print('Size of folds:', folds_size)

    def load_or_create_fold_file(self):
        fold_file_name = '{0:s}/data_split-{1:d}-fold-CV.txt'.format(sys.path[0], self.fold_num)
        if not os.path.exists(fold_file_name):
            fold_entries = {}
            data_list = self.scan_data_source(self.data_source_path)            
            random.shuffle(data_list)
            ptr = 0
            for fold_id in range(self.fold_num-1):
                fold_entries[fold_id] = data_list[ptr:ptr+(len(data_list)//self.fold_num)]
                ptr += len(data_list)//self.fold_num
            fold_entries[self.fold_num-1] = data_list[ptr:]

            with open(fold_file_name, 'w') as fold_file:
                for fold_id in range(self.fold_num):
                    for [case_name, case_path] in fold_entries[fold_id]:
                        fold_file.write('{0:d} {1:s} {2:s}\n'.format(fold_id, case_name, case_path))
        return fold_file_name
    
    def load_data_list(self, fold_file_name):
        exclude_case = []
        fold_data = {}
        with open(fold_file_name, 'r') as fold_file:
            strlines = fold_file.readlines()
            for strline in strlines:
                strline = strline.rstrip('\n')
                params = strline.split()
                fold_id = int(params[0])
                if params[1] in exclude_case:
                    continue
                if fold_id not in fold_data:
                    fold_data[fold_id] = []
                fold_data[fold_id].append([params[1], params[2]])
        return fold_data

    def scan_data_source(self, data_source_path):
        data_list = []
        for casename in os.listdir(data_source_path):
            if not casename.startswith('FL'):
                continue
            case_path = '{0:s}/{1:s}'.format(data_source_path, casename)

            mesh_fn = "{0:s}/mesh.npz".format(case_path)
            hop_dist_fn = "{0:s}/hop_dist.npy".format(case_path)
            mesh_label_fn = "{0:s}/mesh_label.npy".format(case_path)

            if not os.path.exists(mesh_fn) or not os.path.exists(hop_dist_fn) or not os.path.exists(mesh_label_fn):
                print(casename, 'skipped')
                continue
            data_list.append([casename, case_path])
        return data_list

    def get_dataloader_at_fold(self, test_fold_id, test_only=False):
        train_data_list, test_data_list = None, None
        for fold_id in range(self.fold_num):
            if fold_id == test_fold_id:
                test_data_list = self.data_list[fold_id].copy()
            else:
                if train_data_list is None:
                    train_data_list = self.data_list[fold_id].copy()
                else:
                    train_data_list.extend(self.data_list[fold_id])        

        if test_only:
            train_loader = None
        else:
            train_loader = DataListLoader(GraphDataset(train_data_list, enable_augmentation=True, subset_name=self.subset_name), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
        test_loader = DataListLoader(GraphDataset(test_data_list, enable_augmentation=False, subset_name=self.subset_name), batch_size=2, shuffle=False, drop_last=False)

        return train_loader, test_loader

