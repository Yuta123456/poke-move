import torch.nn as nn
from torchvision import models
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import json

from learning.Entity.Pokemon import Pokemon

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
        self.shapes_mapping = json.load(
            "D:/tanaka/Documents/poke-move/data/categorical_mapping/shapes.json"
        )
        self.move_type_mapping = json.load(
            "D:/tanaka/Documents/poke-move/data/categorical_mapping/types.json"
        )
        self.color_mapping = json.load(
            "D:/tanaka/Documents/poke-move/data/categorical_mapping/colors.json"
        )

        self.types_nn = nn.Linear(36, 4)

        self.egg_gropus_nn = nn.Linear(36, 4)

        self.abilities_nn1 = nn.Linear(279 * 3, 256)
        self.abilities_nn2 = nn.Linear(256, 32)
        self.abilities_nn3 = nn.Linear(32, 4)

        self.colors_nn = nn.Linear(10, 4)

        self.shapes_nn = nn.Linear(14, 4)

        self.features_nn1 = nn.Linear(33, 32)
        self.features_nn2 = nn.Linear(32, 64)
        self.features_nn3 = nn.Linear(64, 128)

        self.relu = nn.ReLU()

        self.device = device

    def forward(self, pokemon: Pokemon):
        types_output = self.types_forward(pokemon.types)
        types_output = self.types_nn(types_output)

        egg_groups_output = self.egg_groups_forward(pokemon.egg_groups)
        egg_groups_output = self.egg_gropus_nn(egg_groups_output)

        abilities_output = self.abilities_forward(pokemon.abilities)
        abilities_output = self.abilities_nn1(abilities_output)
        abilities_output = self.relu(abilities_output)
        abilities_output = self.abilities_nn2(abilities_output)
        abilities_output = self.relu(abilities_output)
        abilities_output = self.abilities_nn3(abilities_output)

        colors_output = self.colors_forward(pokemon.color)
        colors_output = self.colors_nn(colors_output)

        shape_output = self.shapes_forward(pokemon.shape)
        shape_output = self.shapes_nn(shape_output)

        other_variables = torch.tensor(
            [
                pokemon.base_experience,
                pokemon.height,
                pokemon.weight,
                pokemon.stats,
                int(pokemon.is_legendary),
                int(pokemon.is_mythical),
                int(pokemon.is_baby),
                pokemon.growth_rate,
            ]
        )
        features = torch.cat(
            # 4, 4, 4, 4, 4, 13
            (
                types_output,
                egg_groups_output,
                abilities_output,
                colors_output,
                shape_output,
                other_variables,
            ),
            dim=1,
        )

    # (batch_size, 36)
    def types_forward(self, types: list[str]):
        types_vector = torch.zeros(18)
        for pokemon_type in types:
            one_hot_vector = torch.nn.functional.one_hot(
                torch.tensor(self.move_type_mapping[pokemon_type])
            )
            types_vector = torch.cat((types_vector, one_hot_vector), dim=1)
        # 単タイプの場合はこの時点で36, 2タイプの場合は48になる
        types_vector = types_vector[-36:]
        return types_vector

    # (batch_size, 30)
    def egg_groups_forward(self, egg_groups: list[str]):
        egg_groups_vector = torch.zeros(15)
        for eg in egg_groups:
            one_hot_vector = torch.nn.functional.one_hot(
                torch.tensor(self.egg_groups_mapping[eg])
            )
            egg_groups_vector = torch.cat((egg_groups_vector, one_hot_vector), dim=1)
        # 単タイプの場合はこの時点で30, 2タイプの場合は45になる
        egg_groups_vector = egg_groups_vector[-30:]
        return egg_groups_vector

    # (batch_size, 279 * 3)
    def abilities_forward(self, abilities: str):
        abilities_vector = torch.zeros(279)
        for ability in abilities:
            one_hot_vector = torch.nn.functional.one_hot(
                torch.tensor(self.abilities_mapping[ability])
            )
            abilities_vector = torch.cat((abilities_vector, one_hot_vector), dim=1)

        abilities_vector = abilities_vector[-279 * 3 :]
        return abilities_vector

    # (batch_size, 10)
    def colors_forward(self, color: str):
        one_hot_vector = torch.nn.functional.one_hot(
            torch.tensor(self.color_mapping[color])
        )
        return one_hot_vector

    # (batch_size, 14)
    def shapes_forward(self, shape: str):
        one_hot_vector = torch.nn.functional.one_hot(
            torch.tensor(self.color_mapping[shape])
        )
        return one_hot_vector
