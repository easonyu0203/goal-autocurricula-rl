import gymnasium as gym
import ale_py
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

tb_log_name = f"ppo_pacman"
model_name = f"ppo_pacman"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from the existing checkpoint if available.")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total number of timesteps to train the model.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    gym.register_envs(ale_py)

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

        env = SubprocVecEnv([_make_env for _ in range(n_envs)])
        env = VecFrameStack(env, n_stack=4)

        return env

    # List of hyperparameters to try
    n_envs = 32
    n_steps = 128

    env = make_env(n_envs=n_envs)
    mini_batch_size = n_steps * n_envs // 4

    # File name of the checkpoint to potentially resume from
    checkpoint_file = f"{model_name}.zip"

    if args.resume_from_checkpoint and os.path.exists(checkpoint_file):
        print(f"Resuming training from checkpoint: {checkpoint_file}")
        ppo = PPO.load(checkpoint_file, env=env, device='cuda')
    else:
        # Creating the PPO model
        ppo = PPO(
            policy=ActorCriticCnnPolicy,
            env=env,
            learning_rate=lambda _: 3e-4,  # Constant learning rate
            n_steps=n_steps,
            batch_size=mini_batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            clip_range_vf=1,
            normalize_advantage=True,
            ent_coef=0.001,
            vf_coef=1,
            target_kl=0.01,
            tensorboard_log='logs',
            device='cuda',
            verbose=1
        )

    # Train the PPO model and log to TensorBoard with a unique name
    tb_log_name = f"PPO_MsPacman_n_envs_{n_envs}_n_steps_{n_steps}"
    ppo.learn(
        total_timesteps=args.total_timesteps,
        tb_log_name=tb_log_name
    )

    # Save the trained model with a unique name
    model_name = f"ppo_mspacman_n_envs_{n_envs}_n_steps_{n_steps}"
    ppo.save(model_name)

if __name__ == "__main__":
    main()