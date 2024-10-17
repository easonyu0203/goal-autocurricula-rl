import gymnasium as gym
from gymnasium.core import ActType, ObsType

class NonEpisodicWrapper(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    def __init__(self, env):
        super(NonEpisodicWrapper, self).__init__(env)

    def step(self, action):
        # Take a step in the environment
        observation, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            # save final observation where user can get it, then reset
            info["terminal_observation"] = observation
            observation, _ = self.env.reset()

        return observation, reward, False, truncated, info # Always return terminated as False

