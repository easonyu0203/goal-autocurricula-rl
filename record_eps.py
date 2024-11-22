import gymnasium as gym
import os

from stable_baselines3.ppo import RNDPPO

model_path = 'output/checkpoints/Walker2d-v4_100000_steps.zip'
video_folder = 'output/videos/Walker2d'


def main():
    # Create the environment with render_mode set to 'rgb_array' for video recording
    env = gym.make("Walker2d-v4", render_mode="rgb_array")

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
    obs, info = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Close the environment
    env.close()

    print(f"Video saved in {video_folder}/")

if __name__ == "__main__":
    main()
