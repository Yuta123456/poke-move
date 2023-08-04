import os
import sys
import pandas as pd
from torch.utils.data import Dataset
import json
import torch

sys.path.append(os.path.abspath("../"))
from learning.Entity.Move import Move

from learning.Entity.Pokemon import Pokemon


class PokemonMoveDataset(Dataset):
    def __init__(self, annotations_file, device):
        self.annotations = pd.read_csv(annotations_file)
        self.device = device
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

        file_path_abilities = (
            "D:/tanaka/Documents/poke-move/data/categorical_mapping/abilities.json"
        )

        file_path_species = (
            "D:/tanaka/Documents/poke-move/data/categorical_mapping/species.json"
        )
        file_path_egg_groups = (
            "D:/tanaka/Documents/poke-move/data/categorical_mapping/egg_groups.json"
        )
        file_path_shapes = (
            "D:/tanaka/Documents/poke-move/data/categorical_mapping/shapes.json"
        )
        file_path_move_type = (
            "D:/tanaka/Documents/poke-move/data/categorical_mapping/types.json"
        )
        file_path_color = (
            "D:/tanaka/Documents/poke-move/data/categorical_mapping/colors.json"
        )

        file_path_move_type = (
            "D:/tanaka/Documents/poke-move/data/categorical_mapping/types.json"
        )
        file_path_damage_class = (
            "D:/tanaka/Documents/poke-move/data/categorical_mapping/damage_class.json"
        )

        with open(file_path_move_type) as f:
            self.move_type_mapping = json.load(f)

        with open(file_path_damage_class) as f:
            self.damage_class_mapping = json.load(f)

        with open(file_path_abilities) as f:
            self.abilities_mapping = json.load(f)

        with open(file_path_species) as f:
            self.species_mapping = json.load(f)

        with open(file_path_egg_groups) as f:
            self.egg_groups_mapping = json.load(f)

        with open(file_path_shapes) as f:
            self.shapes_mapping = json.load(f)

        with open(file_path_move_type) as f:
            self.move_type_mapping = json.load(f)

        with open(file_path_color) as f:
            self.color_mapping = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotion = self.annotations.iloc[idx, :]
        pokemon_id = int(annotion[0])
        move_id = int(annotion[1])
        label = int(annotion[2])
        if label:
            _method = annotion[3]
        pokemon = self.pokemons[pokemon_id - 1]
        move = self.moves[move_id - 1]
        return self.pokemon_to_vector(pokemon), self.move_to_vector(move), label

    def pokemon_to_vector(self, pokemon: Pokemon):
        types_output = self.types_forward(pokemon.types)

        egg_groups_output = self.egg_groups_forward(pokemon.egg_groups)

        abilities_output = self.abilities_forward(pokemon.abilities)

        colors_output = self.colors_forward(pokemon.color)

        shape_output = self.shapes_forward(pokemon.shape)

        # print(
        #     [
        #         pokemon.base_experience,
        #         pokemon.height,
        #         pokemon.weight,
        #         int(pokemon.is_legendary),
        #         int(pokemon.is_mythical),
        #         int(pokemon.is_baby),
        #     ]
        # )
        other_variables = torch.tensor(
            [
                pokemon.base_experience or 0,
                pokemon.height,
                pokemon.weight,
                int(pokemon.is_legendary),
                int(pokemon.is_mythical),
                int(pokemon.is_baby),
            ]
        )
        # print(torch.tensor(pokemon.stats))
        other_variables = torch.cat((other_variables, torch.tensor(pokemon.stats)))
        return {
            "types": types_output.to(self.device),
            "egg_groups": egg_groups_output.to(self.device),
            "abilities": abilities_output.to(self.device),
            "color": colors_output.to(self.device),
            "shape": shape_output.to(self.device),
            "other": other_variables.to(self.device),
        }

    def move_to_vector(self, move: Move):
        description = move.description

        type_output = self.type_forward(move.move_type)
        damage_class_output = self.damage_class_forward(move.damage_class)
        other_variables = torch.tensor(
            [
                move.power or 0,
                move.accuracy or 0,
                move.pp,
                int(move.can_learn_machine),
                move.priority,
            ]
        )

        return {
            "type": type_output.to(self.device),
            "description": description or "",
            "damage_class": damage_class_output.to(self.device),
            "other": other_variables.to(self.device),
        }

    # output (batch_size, 18)
    def type_forward(self, move_type: str):
        type_int = self.move_type_mapping[move_type]
        one_hot_vecrtor_type = torch.nn.functional.one_hot(
            torch.tensor(type_int), num_classes=18
        ).float()
        return one_hot_vecrtor_type

    # output (batch_size, 3)
    def damage_class_forward(self, damage_class: str):
        damage_class_int = self.damage_class_mapping[damage_class]
        one_hot_vecrtor_damage_type = torch.nn.functional.one_hot(
            torch.tensor(damage_class_int), num_classes=3
        ).float()
        return one_hot_vecrtor_damage_type

    # (batch_size, 36)
    def types_forward(self, types: list[str]):
        types_vector = torch.zeros(18).float()
        for pokemon_type in types:
            one_hot_vector = torch.nn.functional.one_hot(
                torch.tensor(self.move_type_mapping[pokemon_type]), num_classes=18
            ).float()
            types_vector = torch.cat((types_vector, one_hot_vector), dim=0)
        # 単タイプの場合はこの時点で36, 2タイプの場合は48になる
        types_vector = types_vector[-36:]
        return types_vector

    # (batch_size, 30)
    def egg_groups_forward(self, egg_groups: list[str]):
        egg_groups_vector = torch.zeros(15).float()
        for eg in egg_groups:
            one_hot_vector = torch.nn.functional.one_hot(
                torch.tensor(self.egg_groups_mapping[eg]), num_classes=15
            ).float()
            egg_groups_vector = torch.cat((egg_groups_vector, one_hot_vector), dim=0)
        # 単タイプの場合はこの時点で30, 2タイプの場合は45になる
        egg_groups_vector = egg_groups_vector[-30:]
        return egg_groups_vector

    # (batch_size, 279 * 3)
    def abilities_forward(self, abilities: str):
        abilities_vector = torch.zeros(837)
        for ability in abilities:
            one_hot_vector = torch.nn.functional.one_hot(
                torch.tensor(self.abilities_mapping[ability]), num_classes=279
            ).float()
            abilities_vector = torch.cat((abilities_vector, one_hot_vector), dim=0)

        abilities_vector = abilities_vector[-279 * 3 :]
        return abilities_vector

    # (batch_size, 10)
    def colors_forward(self, color: str):
        one_hot_vector = torch.nn.functional.one_hot(
            torch.tensor(self.color_mapping[color]),
            num_classes=10,
        ).float()
        return one_hot_vector

    # (batch_size, 14)
    def shapes_forward(self, shape: str):
        if shape == None:
            return torch.zeros(14)
        one_hot_vector = torch.nn.functional.one_hot(
            torch.tensor(self.shapes_mapping[shape]), num_classes=14
        ).float()
        return one_hot_vector
