import torch.nn as nn
from torchvision import models
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import gc


import torch.nn as nn
from torchvision import models
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import gc

if torch.cuda.is_available():
    device = torch.device("cuda")  # GPUデバイスを取得
else:
    device = torch.device("cpu")  # CPUデバイスを取得
device = torch.device("cpu")
import sys
import os

sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("./"))
from learning.Entity.CustomDataset import PokemonMoveDataset


dataset = PokemonMoveDataset(
    "D:/tanaka/Documents/poke-move/data/my-dataset/dataset.txt", device
)

from learning.Encoder.MoveEncoder import MoveEncoder
from learning.Encoder.PokemonEncoder import PokemonEncoder


pokemon_model = PokemonEncoder(device).to(device)
move_model = MoveEncoder(device).to(device)

poke_model_name = "D:/tanaka/Documents/poke-move/learning/pokemon_model_2023-08-04.pth"
move_model_name = "D:/tanaka/Documents/poke-move/learning/move_model_2023-08-04.pth"
pokemon_model.load_state_dict(torch.load(poke_model_name))
move_model.load_state_dict(torch.load(move_model_name))
pokemon_model.eval()
move_model.eval()

# pokemon_vectors = []
# for i, pokemon in enumerate(dataset.pokemons[:5]):
#     poke_vector = dataset.pokemon_to_vector(pokemon)
#     for key in poke_vector.keys():
#         poke_vector[key] = poke_vector[key].unsqueeze(0)
#     v = pokemon_model(poke_vector).to("cpu")
#     pokemon_vectors.append(v)
#     print(f"poke_vector: {sys.getrefcount(poke_vector)}, v: {sys.getrefcount(v)}")
#     for key in poke_vector.keys():
#         print(f"{key}: {sys.getrefcount(poke_vector[key])}")

#     del poke_vector
# pokemon_vectors = torch.cat(pokemon_vectors, dim=0)
from memory_profiler import profile


@profile()
def hoge():
    def generate_move_vectors(dataset):
        for i, move in enumerate(dataset.moves[:20]):
            move = dataset.move_to_vector(move)
            for key in move.keys():
                if key == "description":
                    # move[key] = torch.tensor([move[key]]).to(device)
                    move[key] = [move[key]]
                else:
                    move[key] = move[key].unsqueeze(0)
            with torch.no_grad():
                v = move_model(move).to("cpu")
            print(v.device, v.shape)
            del move
            torch.cuda.empty_cache()
            yield v

            # print(f"move: {sys.getrefcount(move)}, v: {sys.getrefcount(v)}")
            # for key in move.keys():
            #     print(f"{key}: {sys.getrefcount(move[key])}")

    # ジェネレータを使用してmove_vectorsを生成
    move_vectors = []
    for vectors in generate_move_vectors(dataset):
        move_vectors.extend(vectors)


hoge()

print("success")
exit()

len(move_vectors)


import random

move_rankings = []
pokemon_ids = [random.randint(1, 1010) for i in range(5)]
for pokemon_id in pokemon_ids:
    # 各ベクトルのユークリッド距離を計算
    distances = torch.norm(pokemon_vectors[pokemon_id] - move_vectors, dim=1)

    # 距離を用いてベクトルBの添え字を並び替え
    sorted_indices = torch.argsort(distances)
    move_rankings.append(sorted_indices)
text = ""

for i, pokemon_id in enumerate(pokemon_ids):
    pokemon = dataset.pokemons[pokemon_id]
    text += f"選ばれたポケモン: {pokemon.name}\n"
    text += f"近い技 TOP100\n"
    for j, move_id in enumerate(move_rankings[i][:100]):
        move = dataset.moves[move_id]
        text += f"{j}: {move.name}"
