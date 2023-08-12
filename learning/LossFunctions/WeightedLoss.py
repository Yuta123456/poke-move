import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class WeightedLoss(nn.Module):
    def __init__(self, device):
        super(WeightedLoss, self).__init__()
        loss_weight = pd.read_csv(
            "D:/tanaka\Documents\poke-move\learning\LossFunctions\move_loss_weight.csv"
        )
        loss_weight.iloc[:, 1] = loss_weight.iloc[:, 1] / 1225
        self.loss_weight = dict(
            zip(loss_weight.iloc[:, 0], torch.tensor(loss_weight.iloc[:, 1]).to(device))
        )
        self.loss_fn = nn.BCELoss()
        self.device = device

    def forward(self, output1, output2, move_ids):
        loss = self.loss_fn(output1, output2)
        weights = torch.tensor(
            [self.loss_weight[move_id.item()] for move_id in move_ids]
        ).to(self.device)
        loss = loss * weights
        return loss
