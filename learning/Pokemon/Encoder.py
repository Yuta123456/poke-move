import torch.nn as nn
from torchvision import models
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import pandas as pd
"""
画像処理のモデル
"""

# class ImageEncoder(nn.Module):
#     def __init__(self, embedding_size):
#         super(ImageEncoder, self).__init__()
#         self.resnet50 = models.resnet50(pretrained=True)
#         self.fc = nn.Linear(self.resnet50.fc.out_features, embedding_size)
    
#     def forward(self, x):
#         x = self.resnet50(x)
#         x = self.fc(x)
#         return x

class PokemonEncoder():
    