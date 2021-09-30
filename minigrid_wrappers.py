import gym
from gym.spaces import Discrete
import numpy as np
from gym_minigrid.minigrid import Goal
import random


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


class RewardWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        self.env.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.env.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.env.grid.get(*fwd_pos)

        # Rotate left
        if action == self.env.actions.left:
            self.env.agent_dir -= 1
            if self.env.agent_dir < 0:
                self.env.agent_dir += 4

        # Rotate right
        elif action == self.env.actions.right:
            self.env.agent_dir = (self.env.agent_dir + 1) % 4

        # Move forward
        elif action == self.env.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.env.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = 1
            if fwd_cell != None and fwd_cell.type == 'lava':
                reward = -2
                done = True

        # Pick up an object
        elif action == self.env.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.env.carrying is None:
                    self.env.carrying = fwd_cell
                    self.env.carrying.cur_pos = np.array([-1, -1])
                    self.env.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.env.actions.drop:
            if not fwd_cell and self.env.carrying:
                self.env.grid.set(*fwd_pos, self.env.carrying)
                self.env.carrying.cur_pos = fwd_pos
                self.env.carrying = None

        # Toggle/activate an object
        elif action == self.env.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self.env, fwd_pos)

        # Done action (not used by default)
        elif action == self.env.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.env.step_count >= self.env.max_steps:
            done = True

        obs = self.env.gen_obs()
        reward += -0.001
        return obs, reward, done, {}

    def reset(self, random_start=False):
        # Current position and direction of the agent
        self.env.agent_pos = None
        self.env.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self.env._gen_grid(self.env.width, self.env.height)

        # These fields should be defined by _gen_grid
        assert self.env.agent_pos is not None
        assert self.env.agent_dir is not None

        if random_start:
            self.env.agent_pos = (random.randint(1,self.env.width-2),random.randint(1,self.env.height-2))
            start_cell = self.env.grid.get(*self.env.agent_pos)
            while start_cell is not None:
                self.env.agent_pos = (random.randint(1,self.env.width-2),random.randint(1,self.env.height-2))
                start_cell = self.env.grid.get(*self.env.agent_pos)

        # Check that the agent doesn't overlap with an object
        start_cell = self.env.grid.get(*self.env.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.env.carrying = None

        # Step count since episode start
        self.env.step_count = 0

        # Return first observation
        obs = self.env.gen_obs()
        return obs