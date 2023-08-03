import torch.nn as nn
from torchvision import models
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import json

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


class PokemonEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.abilities_mapping = json.load(
            "D:/tanaka/Documents/poke-move/data/categorical_mapping/abilities.json"
        )
        self.species_mapping = json.load(
            "D:/tanaka/Documents/poke-move/data/categorical_mapping/species.json"
        )
        self.egg_groups_mapping = json.load(
            "D:/tanaka/Documents/poke-move/data/categorical_mapping/egg_groups.json"
        )

        self.type_nn = nn.Linear(18, 4)
        self.damage_class_nn = nn.Linear(3, 1)
        self.features_nn1 = nn.Linear(14, 32)
        self.features_nn2 = nn.Linear(32, 64)
        self.features_nn3 = nn.Linear(64, 128)

        self.relu = nn.ReLU()

        self.device = device
