import gymnasium as gym
import ale_py
import os

from stable_baselines3.ppo import RNDPPO
from stable_baselines3.common.policies import RNDActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList


tb_log_name = "MontezumaRevenge-non-episodic"
model_name = "MontezumaRevenge"
n_envs = 28
n_steps = 128
checkpoint_file = None
save_freq = 1_000_000
save_path = "output"
resume_from_checkpoint = False
total_timesteps = 1_000_000_000_000
non_episodic=True
n_frame_stack = 4

def main():
    def make_env(n_envs: int = 1):
        def _make_env():
            env = gym.make("ALE/MontezumaRevenge", render_mode="rgb_array") 
            env = AtariWrapper(env)
            env = Monitor(env)
            return env

        env = SubprocVecEnv([_make_env for _ in range(n_envs)])
        env = VecFrameStack(env, n_stack=n_frame_stack)
        return env

    env = make_env(n_envs=n_envs)
    mini_batch_size = n_steps * n_envs // 4


    if resume_from_checkpoint and os.path.exists(checkpoint_file):
        print(f"Resuming training from checkpoint: {checkpoint_file}")
        ppo = RNDPPO.load(checkpoint_file, env=env, device='cuda')
    else:
        print("Starting training from scratch...")
        # Creating the PPO model
        ppo = RNDPPO(
            policy=RNDActorCriticPolicy,
            policy_kwargs={
                'share_features_extractor': True,
                'normalize_images': True,
                'ortho_init': True,
                'rnd_feature_dim': 512,
                'features_extractor_class': NatureCNN,
            },
            env=env,
            learning_rate=lambda _: 3e-4,  # Constant learning rate
            n_steps=n_steps,
            batch_size=mini_batch_size,
            n_epochs=4,
            non_episodic=non_episodic,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            clip_range_vf=1,
            normalize_advantage=True,
            ent_coef=0.001,
            vf_coef=1,
            rnd_coef=0.1,
            intr_reward_coef=1.0,
            extr_reward_coef=1.0,
            rnd_update_proportion=0.25,
            target_kl=0.01,
            tensorboard_log='logs',
            device='cuda',
            verbose=1
        )

    # Train the PPO model and log to TensorBoard with a unique name
    ppo.learn(
        total_timesteps=total_timesteps,
        tb_log_name=tb_log_name,
        reset_num_timesteps=not resume_from_checkpoint,
        callback=CallbackList([
            CheckpointCallback(save_freq=max(save_freq // n_envs, 1), save_path=os.path.join(save_path, "checkpoints"), name_prefix=model_name, verbose=2)
        ])
    )

    # Save the trained model with a unique name
    ppo.save(os.path.join(save_path, f"{model_name}.zip"))

if __name__ == "__main__":
    main()