import gymnasium as gym
from gymnasium.wrappers.frame_stack import FrameStack
import ale_py
import os
import matplotlib.pyplot as plt
import torch
from stable_baselines3.common.utils import obs_as_tensor

import numpy as np

from stable_baselines3.common.atari_wrappers import AtariWrapper
# stack frames

from stable_baselines3.ppo import RNDPPO

model_path = 'output/checkpoints/MontezumaRevenge_7999488_steps.zip'
video_folder = 'output/videos/MontezumaRevenge'

class MergeFrameWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Update the observation space to reflect the new shape
        obs_shape = self.observation_space.shape  # Original shape
        new_shape = (obs_shape[0] * obs_shape[3], obs_shape[1], obs_shape[2])
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.min(),
            high=self.observation_space.high.max(),
            shape=new_shape,
            dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        # Reorder dimensions from (4, 84, 84, 1) to (4, 1, 84, 84)
        reordered_obs = np.transpose(observation, (0, 3, 1, 2))
        # Merge the first two dimensions to get shape (4, 84, 84)
        merged_obs = reordered_obs.reshape(-1, *reordered_obs.shape[2:])
        return merged_obs



def main():
    # Create the environment with render_mode set to 'rgb_array' for video recording
    env = gym.make("ALE/MontezumaRevenge", render_mode="rgb_array") 
    env = AtariWrapper(env)
    env = FrameStack(env, num_stack=4)
    env = MergeFrameWrapper(env)


    # Specify the directory to save the video
    os.makedirs(video_folder, exist_ok=True)

    # Wrap the environment to record video
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda episode_id: True,  # Record every episode
        disable_logger=True
    )

    # Load the trained model
    model = RNDPPO.load(model_path)

    # Run the model on the environment and record video
    rewards_list = []
    obs, info = env.reset()
    done = False

    while not done:
        with torch.no_grad():
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # comput intrinsic reward
            obs_th = obs_as_tensor(np.array([obs]), device=model.policy.device)
            intr_reward = model.policy.compute_intrinsic_reward(obs_th)
            rewards_list.append(intr_reward.item())

            done = terminated or truncated

    # Close the environment
    env.close()

    # Plot the rewards
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_list)
    plt.xlabel('Step')
    plt.ylabel('Intrinsics Reward')
    plt.title('Intrinsics Rewards over Time')
    plt.grid(True)
    plot_path = os.path.join(video_folder, 'rewards_plot.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Video saved in {video_folder}/")

if __name__ == "__main__":
    main()
