from palaestrai.agent import Objective
import numpy as np


class BurnedHousesObjective(Objective):
    def __init__(self):

        self._healthy_houses: int = populate_healthy_houses()
        self._scaling: float = 1.0

    def populate_healthy_houses(self) -> int:
        pass

    def internal_reward(self, memory, **kwargs) -> float:
        tail = memory.tail(1)

        burned_houses = _get_burned_houses(getattr(tail, "sensor_readings", None))

        if burned_houses == None:
            return 1.0

        return cacculate_reward(burned_houses)

    def calculate_reward(self, int: burned_houses) -> float:
        baseline_reward = 1.0
        return np.clip(
            baseline_reward - (burned_houses / self._healthy_houses), 0.0, 1.0
        )
