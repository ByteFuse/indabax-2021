import gym
from gym.spaces import Discrete
import numpy as np
from gym_minigrid.minigrid import Goal

class DirectionObsWrapper(gym.core.ObservationWrapper):
    """
    Provides the slope/angular direction to the goal with the observations as modeled by (y2 - y2 )/( x2 - x1)
    type = {slope , angle}
    """
    def __init__(self, env,type='angle'):
        super().__init__(env)
        self.goal_position = None
        self.type = type

    def reset(self):
        obs = self.env.reset()
        if not self.goal_position:
            self.goal_position = [x for x,y in enumerate(self.grid.grid) if isinstance(y,(Goal) ) ]
            if len(self.goal_position) >= 1: # in case there are multiple goals , needs to be handled for other env types
                self.goal_position = (int(self.goal_position[0]/self.height) , self.goal_position[0]%self.width)
        return obs

    def observation(self, obs):
        if  self.goal_position[0] - self.agent_pos[0] == 0:
            slope = np.inf
        else:
            slope = np.divide( self.goal_position[1] - self.agent_pos[1] ,  self.goal_position[0] - self.agent_pos[0])
        obs['goal_direction'] = (np.arctan( slope )*(180/np.pi)) if self.type == 'angle' else slope
        return obs['goal_direction']

class FourDirectionsActionWrapper(gym.core.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(4)

    def action(self, action):
        while self.env.agent_dir != action:
            self.env.step(0)        
        return 2


class CoordsObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return self.agent_pos[1], self.agent_pos[0]