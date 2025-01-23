import torch
import time
import os, sys
sys.path.append('../')
from loss_function.dice_loss import DiceLoss
from torch_geometric.nn import DataParallel, aggr
from tqdm import tqdm
import numpy as np

class Trainer():
    def __init__(self, model, dataloader, device_ids):
        self.model = model
        self.device_ids = device_ids
        self.dataloader = dataloader
        self.optimizer = torch.optim.Adam(self.model.module.parameters(), lr=0.001, weight_decay=5e-4)

    def train(self, epoch_num, cp_filename, loss_filename):
        torch.enable_grad()
        self.model.train()
        #class_weight = torch.ones(25).cuda()
        #class_weight[0] = 0.0001
        #class_weight = class_weight / class_weight.sum()
        #loss_func = nn.CrossEntropyLoss(weight=class_weight)
        mesh_loss_func = DiceLoss()
        map_loss_func1 = DiceLoss()
        map_loss_func2 = torch.nn.MSELoss()
        con_loss_func = torch.nn.MSELoss()
        aggr_layer = aggr.MaxAggregation()

        for epoch_id in range(epoch_num):
            epoch_t0 = time.perf_counter()
            epoch_loss = 0
            #for batch in tqdm (self.dataloader, desc="Epoch {0:04d}/{1:04d}".format(epoch_id+1, epoch_num)):
            for batch_id, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                pred, pred_map, feat = self.model(batch)
                label_map = torch.cat([data.dist_map for data in batch]).to(device=pred_map.device)
                label = torch.cat([data.y for data in batch]).to(device=pred.device)
                loss_mesh = mesh_loss_func(pred, label)
                loss_map_dsc = map_loss_func1(pred_map, label_map)
                loss_map_mse = map_loss_func2(pred_map, label_map)

                #'''
                #sim_loss = 0
                #sim_loss_count = 0
                #diff_loss = 0
                #diff_loss_count = 0
                l1_loss = 0
                l1_loss_count = 0
                batch_slicer = 0
                anchor_lmk_feat = None
                lmk_num = label.shape[-1]
                for mesh_id, data in enumerate(batch):
                    #mesh_feat = feat[batch_slicer:batch_slicer+data.num_nodes, :]
                    #aggr_mesh_feat = aggr_layer(x=mesh_feat, index=data.mesh_mask_label.to(device=torch.device('cuda', self.device_ids[0])))
                    aggr_mesh_feat = feat[mesh_id*lmk_num:(mesh_id+1)*lmk_num,:]
                    if anchor_lmk_feat is None:
                        anchor_lmk_feat = aggr_mesh_feat
                        anchor_lmk_feat = torch.nn.functional.normalize(anchor_lmk_feat, p=2.0, dim=1)
                    else:
                        lmk_feature = aggr_mesh_feat
                        lmk_feature = torch.nn.functional.normalize(lmk_feature, p=2.0, dim=1)

                        
                        con_mat = torch.matmul(anchor_lmk_feat, lmk_feature.transpose(0,1))#.view(1,len(data.lmk_inds),len(data.lmk_inds))
                        diag_ele = torch.diag(con_mat)

                        l1_loss += torch.abs(con_mat).mean() + torch.mean(1 - diag_ele)
                        l1_loss_count += 1

                        #sim_loss += torch.mean(1 - diag_ele)
                        #sim_loss_count += 1

                        #lmk_groups = [
                        #    [0,3,5,8,10,12,18,20,22], # left
                        #    [1,4,6,9,11,13,19,21,23], # right
                        #    [2,7,14,15,16,17], # mid
                        #    #[0,8,10,12,20], # left Ag Goi Go Gos RP
                        #    #[1,9,11,13,21], # right Ag Goi Go Gos RP
                        #    #[3,5,18,22], # left Co Cr RMA SIG
                        #    #[4,6,19,23], # right Co Cr RMA SIG
                        #    #[2,7,14,17], # mid B Gn Me Pg
                        #    #[15], # left MF
                        #    #[16], # right MF
                        #]

                        #for lmk_group1_i, lmk_group1 in enumerate(lmk_groups):
                        #    for lmk_group2_i, lmk_group2 in enumerate(lmk_groups):
                        #        if lmk_group1_i == lmk_group2_i:
                        #            continue                                
                        #        diff_loss += torch.abs(con_mat[lmk_group1, :][:, lmk_group2]).sum()
                        #        diff_loss_count += len(lmk_group1) * len(lmk_group2)

                    batch_slicer += data.num_nodes
                #sim_loss = sim_loss / sim_loss_count
                #diff_loss = diff_loss / diff_loss_count
                l1_loss = l1_loss / l1_loss_count
                #'''

                loss = loss_mesh + loss_map_dsc + loss_map_mse + l1_loss
                epoch_loss += loss.item()
                print('Epoch {0:04d}/{1:04d} --- Progress {2:5.2f}% --- Loss: {3:.6f}({4:.6f}/{5:.6f}/{6:.6f}/{7:.6f})'.format(
                        epoch_id+1, epoch_num, 100.0 * batch_id / len(self.dataloader), loss.item(), loss_mesh.item(), loss_map_dsc.item(), loss_map_mse.item(), l1_loss.item()))
                loss.backward()
                self.optimizer.step()
                del batch, pred, pred_map, label, label_map, loss
            epoch_t1 = time.perf_counter()
            epoch_t = epoch_t1 - epoch_t0
            epoch_loss /= len(self.dataloader)
            loss_line = 'Epoch {0:04d}/{1:04d} --- Loss average: {2:.6f} --- time cost: {3:>02d}:{4:>02d}:{5:>02d}\n'.format(epoch_id+1, epoch_num, epoch_loss, int(epoch_t) // 3600, (int(epoch_t) % 3600) // 60, int(epoch_t) % 60)
            print(loss_line)
            with open(loss_filename, 'a') as loss_file:
                loss_file.write(loss_line)
        
        self.save_model(cp_filename)

        return self.model

    def save_model(self, cp_filename):
        model_cp = {}
        model_cp['model_state_dict'] = self.model.state_dict()
        model_cp['optimizer'] = self.optimizer.state_dict()
        torch.save(model_cp, cp_filename)
        print('Trained model saved to:', cp_filename)