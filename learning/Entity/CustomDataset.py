import os
import sys
import pandas as pd
from torch.utils.data import Dataset
import json

sys.path.append(os.path.abspath("../"))
from learning.Entity.Move import Move

from learning.Entity.Pokemon import Pokemon


class PokemonMoveDataset(Dataset):
    def __init__(self, annotations_file):
        self.annotations = pd.read_csv(annotations_file)
        self.pokemons = []
        self.moves = []
        with open(
            "D:/tanaka/Documents/poke-move/data/my-dataset/pokemons.json",
            "r",
            encoding="utf-8",
        ) as f:
            pokemons = json.load(f)
        for p in pokemons:
            pokemon = Pokemon(
                pokemon_id=p["pokemon_id"],
                name=p["name"],
                types=p["types"],
                egg_groups=p["egg_groups"],
                base_experience=p["base_experience"],
                abilities=p["abilities"],
                height=p["height"],
                weight=p["weight"],
                stats=p["stats"],
                color=p["color"],
                shape=p["shape"],
                is_legendary=p["is_legendary"],
                is_baby=p["is_baby"],
                is_mythical=p["is_mythical"],
            )
            self.pokemons.append(pokemon)

        with open(
            "D:/tanaka/Documents/poke-move/data/my-dataset/moves.json",
            "r",
            encoding="utf-8",
        ) as f:
            moves = json.load(f)
        for m in moves:
            move = Move(
                move_id=m["move_id"],
                name=m["name"],
                move_type=m["move_type"],
                description=m["description"],
                accuracy=m["accuracy"],
                damage_class=m["damage_class"],
                power=m["power"],
                pp=m["pp"],
                priority=m["priority"],
                can_learn_machine=m["can_learn_machine"],
            )
            self.moves.append(move)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotion = self.annotations.iloc[idx, :]
        pokemon_id = int(annotion[0])
        move_id = int(annotion[1])
        label = int(annotion[2])
        if label:
            _method = annotion[3]

        return self.pokemons[pokemon_id - 1], self.moves[move_id - 1], label
