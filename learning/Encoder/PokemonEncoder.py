import torch.nn as nn
from torchvision import models
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import json

from learning.Entity.Pokemon import Pokemon


class PokemonEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.types_nn = nn.Linear(36, 4)

        self.egg_gropus_nn = nn.Linear(30, 4)

        self.abilities_nn1 = nn.Linear(279 * 3, 256)
        self.abilities_nn2 = nn.Linear(256, 32)
        self.abilities_nn3 = nn.Linear(32, 4)

        self.colors_nn = nn.Linear(10, 4)

        self.shapes_nn = nn.Linear(14, 4)

        self.features_nn1 = nn.Linear(32, 32)
        self.features_nn2 = nn.Linear(32, 64)
        self.features_nn3 = nn.Linear(64, 128)

        self.relu = nn.ReLU()

        self.device = device

    def forward(self, pokemon):
        types_output = self.types_nn(pokemon["types"])

        egg_groups_output = self.egg_gropus_nn(pokemon["egg_groups"])

        abilities_output = self.abilities_nn1(pokemon["abilities"])
        abilities_output = self.relu(abilities_output)
        abilities_output = self.abilities_nn2(abilities_output)
        abilities_output = self.relu(abilities_output)
        abilities_output = self.abilities_nn3(abilities_output)

        colors_output = self.colors_nn(pokemon["color"])

        shape_output = self.shapes_nn(pokemon["shape"])
        features = torch.cat(
            # 4, 4, 4, 4, 4, 13
            (
                types_output,
                egg_groups_output,
                abilities_output,
                colors_output,
                shape_output,
                pokemon["other"],
            ),
            dim=1,
        )
        features = self.features_nn1(features)
        features = self.relu(features)
        features = self.features_nn2(features)
        features = self.relu(features)
        features = self.features_nn3(features)
        return features
