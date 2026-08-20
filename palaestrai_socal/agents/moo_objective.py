from palaestrai.agent.objective import Objective

from palaestrai_socal.agents.saidi_objective import SaidiObjective
from palaestrai_socal.agents.hff_objective import HealthyFirefighterObjective
from palaestrai_socal.agents.buho_objective import BurnedHousesObjective

class MooObjective(Objective):
    def __init__(
            self,
            alpha: float = 1/3,
            beta: float = 1/3,
            gamma: float = 1/3
        ):

        settings = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma
        }
        
        super().__init__(params=settings)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def internal_reward(self, memory, **kwargs) -> float:

        saidi = SaidiObjective(memory, **kwargs).internal_reward(memory, **kwargs)
        healthy_firefighter = HealthyFirefighterObjective().internal_reward(memory, **kwargs)
        burned_houses = BurnedHousesObjective().internal_reward(memory, **kwargs)

        return saidi*self.alpha + healthy_firefighter*self.beta + burned_houses*self.gamma




        





    