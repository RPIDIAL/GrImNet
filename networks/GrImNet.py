import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tg_nn

class DoubleConvNormNonlinBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, first_stride):
        super(DoubleConvNormNonlinBlock, self).__init__()
        padding = [1 if i == 3 else 0 for i in kernel_size]
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, stride=first_stride, padding=padding),
            nn.InstanceNorm3d(out_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size, stride=[1, 1, 1], padding=padding),
            nn.InstanceNorm3d(out_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        y = self.conv(x)
        return y

class Upsampler(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsampler, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)

class GrImNet(nn.Module):
    def __init__(self, geo_feat_dim, hidden_dim, lmk_num):
        super(GrImNet, self).__init__()

        self.encoding_path = []
        self.encoding_path.append(DoubleConvNormNonlinBlock(1,        32,     kernel_size=[3,3,3], first_stride=[1,1,1]))
        self.encoding_path.append(DoubleConvNormNonlinBlock(32,       64,     kernel_size=[3,3,3], first_stride=[2,2,2]))
        self.encoding_path.append(DoubleConvNormNonlinBlock(64,       128,    kernel_size=[3,3,3], first_stride=[2,2,2]))
        self.encoding_path.append(DoubleConvNormNonlinBlock(128,      256,    kernel_size=[3,3,3], first_stride=[2,2,2]))
        self.encoding_path.append(DoubleConvNormNonlinBlock(256,      320,    kernel_size=[3,3,3], first_stride=[2,2,2]))
        self.encoding_path = nn.ModuleList(self.encoding_path)

        self.decoding_path = []
        self.decoding_path.append(DoubleConvNormNonlinBlock(512,      256,    kernel_size=[3,3,3], first_stride=[1,1,1]))
        self.decoding_path.append(DoubleConvNormNonlinBlock(256,      128,    kernel_size=[3,3,3], first_stride=[1,1,1]))
        self.decoding_path.append(DoubleConvNormNonlinBlock(128,      64,     kernel_size=[3,3,3], first_stride=[1,1,1]))
        self.decoding_path.append(DoubleConvNormNonlinBlock(64,       32,     kernel_size=[3,3,3], first_stride=[1,1,1]))
        self.decoding_path = nn.ModuleList(self.decoding_path)

        self.upsamplers = []
        self.upsamplers.append(nn.ConvTranspose3d(320,  256,    [2,2,2],    stride=[2,2,2],     bias=False))
        self.upsamplers.append(nn.ConvTranspose3d(256,  128,    [2,2,2],    stride=[2,2,2],     bias=False))
        self.upsamplers.append(nn.ConvTranspose3d(128,  64,     [2,2,2],    stride=[2,2,2],     bias=False))
        self.upsamplers.append(nn.ConvTranspose3d(64,   32,     [2,2,2],    stride=[2,2,2],     bias=False))
        self.upsamplers = nn.ModuleList(self.upsamplers)

        self.output_convs = []
        self.output_convs.append(nn.Conv3d(256,   lmk_num,    [1,1,1],    stride=[1,1,1],     bias=False))
        self.output_convs.append(nn.Conv3d(128,   lmk_num,    [1,1,1],    stride=[1,1,1],     bias=False))
        self.output_convs.append(nn.Conv3d(64,    lmk_num,    [1,1,1],    stride=[1,1,1],     bias=False))
        self.output_convs.append(nn.Conv3d(32,    lmk_num,    [1,1,1],    stride=[1,1,1],     bias=False))
        self.output_convs = nn.ModuleList(self.output_convs)

        self.output_nonlin = lambda x: torch.sigmoid(x)

        self.conv1 = tg_nn.GraphConv(geo_feat_dim + lmk_num, hidden_dim, aggr='add')
        self.conv2 = tg_nn.GraphConv(hidden_dim, hidden_dim, aggr='add')
        self.conv3 = tg_nn.GraphConv(hidden_dim, hidden_dim, aggr='max')
        self.conv4 = tg_nn.SAGEConv(hidden_dim, hidden_dim, aggr='max')
        self.conv5 = tg_nn.SAGEConv(hidden_dim, hidden_dim, aggr='max')
        self.conv6 = tg_nn.SAGEConv(hidden_dim, hidden_dim, aggr='max')

        self.jk1 = tg_nn.JumpingKnowledge("lstm", hidden_dim, 3)
        self.jk2 = tg_nn.JumpingKnowledge("lstm", hidden_dim, 3)

        self.aggr_layer = tg_nn.aggr.MaxAggregation()

        self.lin1 = nn.Linear(hidden_dim, 63)
        self.lin2 = nn.Linear(63, lmk_num)

        self.active1 = nn.PReLU(hidden_dim)
        self.active2 = nn.PReLU(hidden_dim)
        self.active3 = nn.PReLU(hidden_dim)
        self.active4 = nn.PReLU(hidden_dim)
        self.active5 = nn.PReLU(hidden_dim)
        self.active6 = nn.PReLU(hidden_dim)
        self.active7 = nn.PReLU(63)
        self.active8 = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, edge_weight, grid_pos, dist_map = data.x, data.edge_index, data.edge_weight, data.grid_pos, data.dist_map

        im_x, batch_size = data.image_data, data.num_graphs

        assert batch_size == 1

        skips = []

        for i in range(len(self.encoding_path) - 1):
            im_x = self.encoding_path[i](im_x)
            skips.append(im_x)

        im_x = self.encoding_path[-1](im_x)

        for i in range(len(self.upsamplers)):
            im_x = self.upsamplers[i](im_x)
            im_x = torch.cat((im_x, skips[-(1 + i)]), dim=1)
            im_x = self.decoding_path[i](im_x)
        
        im_out = self.output_nonlin(self.output_convs[-1](im_x))

        im_feat = F.grid_sample(im_out, grid_pos)
        im_feat = im_feat.squeeze().transpose(0,1)
        gt_feat = None
        if self.training:
            gt_feat = F.grid_sample(dist_map, grid_pos)
            gt_feat = gt_feat.squeeze().transpose(0,1)

        x = torch.cat([im_feat, x], dim=1)
        x = self.conv1(x, edge_index, edge_weight)
        x = self.active1(x)
        xs = [x]

        x = self.conv2(x, edge_index, edge_weight)
        x = self.active2(x)
        xs += [x]

        x = self.conv3(x, edge_index, edge_weight)
        x = self.active3(x)
        xs += [x]

        # ~~~~~~~~~~~~Jumping knowledge applied ~~~~~~~~~~~~~~~
        x = self.jk1(xs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        x = self.conv4(x, edge_index)
        x = self.active4(x)
        xs = [x]

        x = self.conv5(x, edge_index)
        x = self.active5(x)
        xs += [x]

        x = self.conv6(x, edge_index)
        x = self.active6(x)
        xs += [x]

        # ~~~~~~~~~~~~Jumping knowledge applied ~~~~~~~~~~~~~~~
        x = self.jk2(xs)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        lmk_feat = None
        if self.training:
            im_feat_max, im_feat_max_id = torch.max(gt_feat, dim=1)
            im_feat_max_id += 1
            im_feat_max_id[im_feat_max < 0.1] = 0
        
            lmk_feat = self.aggr_layer(x=x, index=im_feat_max_id)
            lmk_feat = lmk_feat[1:, :]
        
        x = self.lin1(x)
        x = self.active7(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin2(x)
        x = self.active8(x)

        if self.training:
            return x, im_out, lmk_feat
        else:
            return x, im_out