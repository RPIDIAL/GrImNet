import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, ignore_label=None):
        if ignore_label is None:
            intersection = torch.sum(input * target)
            union = torch.sum(input**2 + target**2)
        else:
            input[target==ignore_label] = 0
            masked_target = target.detach()
            masked_target[target==ignore_label] = 0
            intersection = torch.sum(input * masked_target)
            union = torch.sum(input**2 + masked_target**2)
        smooth = 1e-6
        dsc = (2*intersection + smooth) / (union + smooth)
        return 1.0-dsc

