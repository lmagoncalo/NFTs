import torch
from torch import nn


class SymmetryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_loss(self, out):
        mseloss = nn.MSELoss()
        cur_loss = mseloss(out, torch.flip(out, [3]))
        return cur_loss
