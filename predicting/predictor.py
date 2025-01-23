import torch
import time
import os, sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm

import SimpleITK as sitk

class Predictor():
    def __init__(self, model, testloader, trainloader, device_ids):
        self.model = model
        self.device_ids = device_ids
        self.testloader = testloader
        self.trainloader = trainloader

    def predict(self, result_path):
        os.makedirs(result_path, exist_ok=True)
        torch.no_grad()
        self.model.eval()
        '''
        atlas_feature = {}
        #atlas_feature = None
        #atlas_num = 0
        for batch in tqdm(self.trainloader, desc="Generating atlas features ... "):
            _, feat = self.model(batch)
            batch_slicer = 0
            for i, data in enumerate(batch):
                case_feat = feat[batch_slicer:batch_slicer+data.num_nodes, :].detach()
                atlas_feature[data.casename] = case_feat[data.lmk_inds,:].detach()
                #if atlas_feature is None:
                #    atlas_feature = case_feat[data.lmk_inds,:].detach()
                #else:
                #    atlas_feature += case_feat[data.lmk_inds,:].detach()
                #atlas_num += 1
                batch_slicer += data.num_nodes
            del batch, feat
        #atlas_feature = atlas_feature / atlas_num
        '''

        for batch in tqdm(self.testloader, desc="Testing ... "):
            pred, _ = self.model(batch)                    
            batch_slicer = 0
            for i, data in enumerate(batch):
                '''
                test_feat = feat[batch_slicer:batch_slicer+data.num_nodes, :].detach()

                votes = torch.zeros([len(data.lmk_inds), data.num_nodes], dtype=int)
                for casename in atlas_feature:
                    cormat = torch.matmul(atlas_feature[casename],test_feat.transpose(0,1))
                    lmk_inds = torch.argmax(cormat, dim=1)
                    for lmk_id, lmk_ind in enumerate(lmk_inds):
                        votes[lmk_id][lmk_ind] += 1
                lmk_votes = torch.max(votes, dim=1)
                lmk_inds = torch.argmax(votes, dim=1)

                #cormat = torch.matmul(atlas_feature,test_feat.transpose(0,1))
                #lmk_inds = torch.argmax(cormat, dim=1)
                lmk_coord = data.points[lmk_inds,:]

                #for i in range(lmk_coord.shape[0]):
                #    with open('{0:s}/{1:s}.txt'.format(result_path, data.casename), 'a') as case_file:
                #        case_file.write('{0:d} {1:.6f} {2:.6f} {3:.6f} {4:.6f}\n'.format(i, lmk_coord[i][0], lmk_coord[i][1], lmk_coord[i][2], 1.0))
                '''

                '''
                lmk_ind = torch.argmax(pred[i,:,:,:,:].view(pred.shape[1], -1), dim=1).detach().cpu().numpy()
                lmk_ind_z, lmk_ind_y, lmk_ind_x = np.unravel_index(lmk_ind, (pred.shape[2], pred.shape[3], pred.shape[4]))
                lmk_ind = np.concatenate((lmk_ind_x.reshape(-1,1), lmk_ind_y.reshape(-1,1), lmk_ind_z.reshape(-1,1)), axis=1)
                lmk_coord = lmk_ind * data.image_spacing + data.image_origin
                for i in range(lmk_coord.shape[0]):
                    with open('{0:s}/{1:s}.txt'.format(result_path, data.casename), 'a') as case_file:
                        case_file.write('{0:d} {1:.6f} {2:.6f} {3:.6f} {4:.6f}\n'.format(i, lmk_coord[i][0], lmk_coord[i][1], lmk_coord[i][2], 1.0))
                '''
                np.save('{0:s}/{1:s}.npy'.format(result_path, data.casename), pred[batch_slicer:batch_slicer+data.num_nodes, :].detach().cpu().numpy())

                #if data.casename == 'FL022':
                #    tmp_fn = '{0:s}/{1:s}-im.nii.gz'.format(result_path, data.casename)
                #    sitk.WriteImage(sitk.GetImageFromArray(data.image_data[0,0,:].detach().cpu().numpy()), tmp_fn)
                #    for c in range(lmk_map.shape[1]):
                #        tmp_fn = '{0:s}/{1:s}-{2:d}.nii.gz'.format(result_path, data.casename, c)
                #        sitk.WriteImage(sitk.GetImageFromArray(lmk_map[i,c,:].detach().cpu().numpy()), tmp_fn)
                #        tmp_fn = '{0:s}/{1:s}-{2:d}-gt.nii.gz'.format(result_path, data.casename, c)
                #        sitk.WriteImage(sitk.GetImageFromArray(data.dist_map[0,c,:].detach().cpu().numpy()), tmp_fn)

                batch_slicer += data.num_nodes
            del batch, pred#, feat