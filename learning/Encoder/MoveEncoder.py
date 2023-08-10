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

        # self.tokenizer = AutoTokenizer.from_pretrained("YituTech/conv-bert-base")
        # self.bert = AutoModel.from_pretrained("YituTech/conv-bert-base")
        # self.bert_nn1 = nn.Linear(768, 256)
        # self.bert_nn2 = nn.Linear(256, 32)
        # self.bert_nn3 = nn.Linear(32, 4)

        self.type_nn = nn.Linear(18, 4)
        self.damage_class_nn = nn.Linear(3, 1)

        # self.features_nn1 = nn.Linear(14, 32)
        self.features_nn1 = nn.Linear(10, 32)
        self.features_nn2 = nn.Linear(32, 64)
        self.features_nn3 = nn.Linear(64, 128)

        self.relu = nn.ReLU()

        self.device = device

    def forward(self, move):
        # print(move)
        # bert_output = self.description_forward(move["description"])
        # bert_output = self.bert_nn1(bert_output)
        # bert_output = self.relu(bert_output)
        # bert_output = self.bert_nn2(bert_output)
        # bert_output = self.relu(bert_output)
        # bert_output = self.bert_nn3(bert_output)
        # bert_output = self.relu(bert_output)

        type_output = move["type"]
        type_output = self.type_nn(type_output)
        damage_class_output = move["damage_class"]
        damage_class_output = self.damage_class_nn(damage_class_output)
        features = torch.cat(
            # 4, 4, 1, 5
            # (bert_output, type_output, damage_class_output, move["other"]),
            (type_output, damage_class_output, move["other"]),
            dim=1,
        )

        features = self.features_nn1(features)
        features = self.relu(features)
        features = self.features_nn2(features)
        features = self.relu(features)
        output = self.features_nn3(features)
        # output = self.relu(output)
        del features, type_output, damage_class_output
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
        bert_output = self.bert(ids)
        bert_output = bert_output.last_hidden_state[:, 0, :]
        del ids
        return bert_output
