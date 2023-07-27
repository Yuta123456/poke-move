import torch.nn as nn
from torchvision import models
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import pandas as pd

class MoveEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese-v2")

    def forward(self, x):
        x = self.bert(x)
        x = x.last_hidden_state
        # print(x.shape)
        x = x[:,0,:] 
        # print(x.shape)
        return x