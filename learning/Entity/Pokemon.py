class Pokemon:
    def __init__(
        self,
        pokemon_id: int,
        name: str,
        types: list[str],
        egg_groups: list[str],
        base_experience: int,
        abilities: list[str],
        height: int,
        weight: int,
        stats: list[int],
        color: str,
        shape: str,
        is_legendary: bool,
        is_mythical: bool,
        is_baby: bool,
        # growth_rate: int,
    ):
        self.pokemon_id = pokemon_id
        self.name = name
        self.types = types
        self.egg_groups = egg_groups
        self.base_experience = base_experience
        self.abilities = abilities
        self.height = height
        self.weight = weight
        self.stats = stats
        self.color = color
        self.shape = shape
        self.is_legendary = is_legendary
        self.is_mythical = is_mythical
        self.is_baby = is_baby
        # self.growth_rate = growth_rate
