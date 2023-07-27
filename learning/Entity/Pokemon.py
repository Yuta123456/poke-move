class Pokemon:
    def __init__(
        self,
        pokemon_id: int,
        name: str,
        types: list[str],
        egg_groups: str,
        base_experience: int,
        abilities: str,
    ):
        self.pokemon_id = pokemon_id
        self.name = name
        self.types = types
        self.egg_groups = egg_groups
        self.base_experience = base_experience
        self.abilities = abilities
