import os
import gymnasium as gym
import ale_py
from gymnasium.wrappers.resize_observation import ResizeObservation
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from gymnasium.wrappers.frame_stack import FrameStack

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Step 1: Create the Environment
def make_env(n_envs: int = 1):
    def _make_env():
        env = gym.make(
            'ALE/MsPacman-v5',
            obs_type='rgb',
            render_mode="rgb_array",
            mode=0
        )
        env = AtariWrapper(
            env,
            noop_max=250,
            frame_skip=5,
            screen_size=64,
            terminal_on_life_loss=True,
            clip_reward=True,
            action_repeat_probability=0.0,
            grayscale_obs=False,
        )
        env = Monitor(env)
        return env

    env = DummyVecEnv([_make_env for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)

    return env

# Step 2: Create and Wrap the Environment
env = make_env(n_envs=1)

# Step 3: Setup the video folder
video_folder = "./videos/"
os.makedirs(video_folder, exist_ok=True)

# Step 5: Load the trained model
model = PPO.load("ppo_pacman.zip", env=env)

# Step 6: Run a single episode and record it
def run_single_episode(env, model):
    """Run a single episode in the environment and record it."""
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

# Run the single episode and record it
run_single_episode(env, model)
