class Move:
    def __init__(
        self,
        move_id: int,
        name: str,
        move_type: str,
        description: str,
        accuracy: int,
        damage_class: str,
        power: int,
        pp: int,
        priority: int,
        can_learn_machine: bool,
        # stat_changes,
    ):
        self.move_id = move_id
        self.name = name
        self.move_type = move_type
        self.description = description
        self.accuracy = accuracy
        self.damage_class = damage_class
        self.power = power
        self.pp = pp
        self.priority = priority
        self.can_learn_machine = can_learn_machine
        # "stat_changes": [
        #           {
        #       "change": 2,
        #       "stat": {
        #         "name": "attack",
        #         "url": "https://pokeapi.co/api/v2/stat/2/"
        #       }
        #     }
        #   ],
        # self.stat_changes = stat_changes
