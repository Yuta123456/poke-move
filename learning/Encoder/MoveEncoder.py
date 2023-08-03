import torch.nn as nn
from torchvision import models
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import pandas as pd

from learning.Entity.Move import Move
import json


class MoveEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.move_type_mapping = json.load(
            "D:/tanaka/Documents/poke-move/data/categorical_mapping/types.json"
        )
        self.damage_class_mapping = json.load(
            "D:/tanaka/Documents/poke-move/data/categorical_mapping/types.json"
        )

        self.tokenizer = AutoTokenizer.from_pretrained("YituTech/conv-bert-base")
        self.bert = AutoModel.from_pretrained("YituTech/conv-bert-base")
        self.bert_nn1 = nn.Linear(768, 256)
        self.bert_nn2 = nn.Linear(256, 32)
        self.bert_nn3 = nn.Linear(32, 4)

        self.type_nn = nn.Linear(18, 4)
        self.damage_class_nn = nn.Linear(3, 1)

        self.features_nn1 = nn.Linear(14, 32)
        self.features_nn2 = nn.Linear(32, 64)
        self.features_nn3 = nn.Linear(64, 64)

        self.device = device

    def forward(self, move: Move):
        bert_output = self.description_forward(move.description)
        bert_output = self.bert_nn1(bert_output)
        bert_output = self.bert_nn2(bert_output)
        bert_output = self.bert_nn3(bert_output)

        type_output = self.type_forward(move.move_type)
        damage_class_output = self.damage_class_forward(move.damage_class)
        other_variables = torch.tensor(
            [
                move.power,
                move.accuracy,
                move.pp,
                int(move.can_learn_machine),
                move.priority,
            ]
        )
        features = torch.cat(
            # 4, 4, 1, 5
            (bert_output, type_output, damage_class_output, other_variables),
            dim=1,
        )

        features = self.features_nn1(features)
        features = self.features_nn2(features)
        output = self.features_nn3(features)
        return output

    def description_forward(self, description: str):
        ids = self.tokenizer.batch_encode_plus(
            description,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )
        ids = ids.to(self.device)
        bert_output = self.bert(ids).last_hidden_state[:, 0, :]
        return bert_output

    # output (batch_size, 18)
    def type_forward(self, move_type: str):
        type_int = self.move_type_mapping[move_type]
        one_hot_vecrtor_type = torch.nn.functional.one_hot(torch.tensor(type_int))
        return one_hot_vecrtor_type

    # output (batch_size, 3)
    def damage_class_forward(self, damage_class: str):
        damage_class_int = self.damage_class_mapping[damage_class]
        one_hot_vecrtor_damage_type = torch.nn.functional.one_hot(
            torch.tensor(damage_class_int)
        )
        return one_hot_vecrtor_damage_type
