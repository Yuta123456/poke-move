import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import numpy as np
from PIL import Image
import json
from learning.Entity.Move import Move

from learning.Entity.Pokemon import Pokemon


class PokemonMoveDataset(Dataset):
    def __init__(self, annotations_file):
        self.annotations = pd.read_csv(annotations_file)
        self.pokemons = []
        self.moves = []
        for pokemon_id in (1, 1011):
            with open(
                f"D:/tanaka/Documents/poke-move/data/pokemons/{pokemon_id}.json", "r"
            ) as f:
                pokemon_original = json.load(f)
            with open(
                f"D:/tanaka/Documents/poke-move/data/pokemon-species/{pokemon_id}.json",
                "r",
            ) as f:
                pokemon_species = json.load(f)
            types = [t["types"]["name"] for t in pokemon_original["types"]]
            egg_groups = [e["name"] for e in pokemon_species["egg_groups"]]
            abilities = [a["ability"]["name"] for a in pokemon_original["abilities"]]
            stats = [s["base_stat"] for s in pokemon_original["stats"]]
            pokemon = Pokemon(
                pokemon_id=pokemon_id,
                name=pokemon_original["name"],
                types=types,
                egg_groups=egg_groups,
                base_experience=pokemon_original["base_experience"],
                abilities=abilities,
                height=pokemon_original["height"],
                weight=pokemon_original["weight"],
                stats=stats,
                color=pokemon_species["color"]["name"],
                shape=pokemon_species["shape"]["name"],
                is_legendary=pokemon_species["is_legendary"],
                is_baby=pokemon_species["is_baby"],
                is_mythical=pokemon_species["is_mythical"],
            )
            self.pokemons.append(pokemon)
        for move_id in range(1, 901):
            with open(
                f"D:/tanaka/Documents/poke-move/data/moves/{move_id}.json", "r"
            ) as f:
                move_original = json.load(f)
                move = Move(
                    move_id=move_id,
                    name=move_original["name"],
                    move_type=move_original["type"]["name"],
                    description=move_original["flavor_text_entries"],
                )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        return
