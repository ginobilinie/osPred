import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    """
    均方根误差
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted, target):
        mse = self.mse_loss(predicted, target)
        rmse = torch.sqrt(mse)
        return rmse
