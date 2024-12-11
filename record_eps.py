import gymnasium as gym
from gymnasium.wrappers.frame_stack import FrameStack
import ale_py
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from stable_baselines3.common.utils import obs_as_tensor
import numpy as np
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.ppo import RNDPPO
import matplotlib.gridspec as gridspec

model_path = '/hdd/eyu/output/MontezumaRevenge-non-episodic-only-intr-bigger/checkpoints/MontezumaRevenge_670000000_steps.zip'
video_folder = 'output/videos/MontezumaRevenge-non-episodic-only-intr-bigger'
file_name = 'eps_670M_first_reward.mp4'

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
    env = AtariWrapper(env, action_repeat_probability=0, frame_skip=1, terminal_on_life_loss=False)
    env = FrameStack(env, num_stack=4)
    env = MergeFrameWrapper(env)

    # Specify the directory to save the video
    os.makedirs(video_folder, exist_ok=True)

    # Load the trained model
    model = RNDPPO.load(model_path)

    # Run the model on the environment and collect frames and rewards
    rewards_list = []
    frames = []
    obs, info = env.reset()
    done = False

    while not done:
        with torch.no_grad():
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            # Compute intrinsic reward
            obs_th = obs_as_tensor(np.array([obs]), device=model.policy.device)
            obs_th = obs_th[:, -1:, :, :]
            intr_reward = model.policy.compute_intrinsic_reward(obs_th)
            rewards_list.append(intr_reward.item())

            # Capture the current frame
            frames.append(env.render())

            done = terminated or truncated

    # Close the environment
    env.close()

    # Create a dual-panel animation with matplotlib
    fig = plt.figure(figsize=(16, 6))  # Increased width for wider reward plot
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    ax1 = plt.subplot(gs[0])  # For gameplay
    ax2 = plt.subplot(gs[1])  # For reward plot

    # Set up the gameplay axis
    ax1.axis('off')
    img_display = ax1.imshow(frames[0])

    # Set up the reward plot axis
    ax2.set_xlim(0, len(rewards_list))
    ax2.set_ylim(min(rewards_list), max(rewards_list))
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Intrinsic Reward')
    ax2.set_title('Intrinsic Rewards over Time')
    line, = ax2.plot([], [], lw=2)

    def update(frame_idx):
        # Update the gameplay frame
        img_display.set_data(frames[frame_idx])
        
        # Update the rewards plot
        line.set_data(range(frame_idx + 1), rewards_list[:frame_idx + 1])
        return img_display, line

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True, interval=50, repeat=False)

    # Save the animation as a video file
    output_video_path = os.path.join(video_folder, file_name)
    ani.save(output_video_path, writer='ffmpeg')

    plt.close()
    print(f"episode rollout video saved in {output_video_path}")

if __name__ == "__main__":
    main()
