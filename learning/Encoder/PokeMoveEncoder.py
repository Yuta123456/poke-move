import torch.nn as nn
from torchvision import models
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import json

from learning.Entity.Pokemon import Pokemon


class PokeMoveEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("YituTech/conv-bert-base")
        self.bert = AutoModel.from_pretrained("YituTech/conv-bert-base")
        self.bert_nn1 = nn.Linear(768, 256)
        self.bert_nn2 = nn.Linear(256, 32)
        self.bert_nn3 = nn.Linear(32, 4)

        self.types_nn = nn.Linear(36, 4)

        self.egg_gropus_nn = nn.Linear(30, 4)

        self.abilities_nn1 = nn.Linear(279 * 3, 256)
        self.abilities_nn2 = nn.Linear(256, 32)
        self.abilities_nn3 = nn.Linear(32, 4)

        self.colors_nn = nn.Linear(10, 4)

        self.shapes_nn = nn.Linear(14, 4)

        self.p_features_nn1 = nn.Linear(32, 32)
        self.p_features_nn2 = nn.Linear(32, 64)
        self.p_features_nn3 = nn.Linear(64, 64)

        self.type_nn = nn.Linear(18, 4)
        self.damage_class_nn = nn.Linear(3, 1)

        self.m_features_nn1 = nn.Linear(14, 32)
        self.m_features_nn2 = nn.Linear(32, 64)
        self.m_features_nn3 = nn.Linear(64, 64)

        self.nn1 = nn.Linear(128, 64)
        self.nn2 = nn.Linear(64, 16)
        self.nn2 = nn.Linear(64, 16)
        self.nn3 = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.device = device

    def forward(self, pokemon, move):
        bert_output = self.description_forward(move["description"])

        bert_output = self.bert_nn1(bert_output)
        bert_output = self.relu(bert_output)
        bert_output = self.bert_nn2(bert_output)
        bert_output = self.relu(bert_output)
        bert_output = self.bert_nn3(bert_output)
        bert_output = self.relu(bert_output)

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
        features = self.p_features_nn1(features)
        features = self.relu(features)
        features = self.p_features_nn2(features)
        features = self.relu(features)
        pokemon_output = self.p_features_nn3(features)

        type_output = move["type"]
        type_output = self.type_nn(type_output)
        damage_class_output = move["damage_class"]
        damage_class_output = self.damage_class_nn(damage_class_output)
        features = torch.cat(
            # 4, 4, 1, 5
            # (bert_output, type_output, damage_class_output, move["other"]),
            (bert_output, type_output, damage_class_output, move["other"]),
            dim=1,
        )

        features = self.m_features_nn1(features)
        features = self.relu(features)
        features = self.m_features_nn2(features)
        features = self.relu(features)
        move_output = self.m_features_nn3(features)

        features = torch.cat((pokemon_output, move_output), dim=1)
        output = self.nn1(features)
        output = self.relu(output)
        output = self.nn2(output)
        output = self.relu(output)
        output = self.nn3(output)
        output = self.sigmoid(output).view(-1)
        return output

    def description_forward(self, description: str):
        ids = self.tokenizer.batch_encode_plus(
            description,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        ).input_ids
        ids = ids.to(self.device)
        with torch.no_grad():
            bert_output = self.bert(ids)
        bert_output = bert_output.last_hidden_state[:, 0, :]
        return bert_output
