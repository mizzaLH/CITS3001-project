from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from nes_py.wrappers import JoypadSpace
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sys import argv

# Taken from https://github.com/nicknochnack/MarioRL/blob/main/Mario%20Tutorial.ipynb
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)

        return True

# Setup environment
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# Preprocess environment
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env]) # type: ignore
env = VecFrameStack(env, 4, channels_order="last")

# Run training or model based on CLI value
# If value given then run that model

# Create model and logger

# logger = TrainAndLoggingCallback(check_freq=100000, save_path="./models")
# model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.00001, n_steps=512, tensorboard_log="./logs")
#
# model.learn(total_timesteps=10000000, callback=logger)

# Load model from CLI
model = PPO.load("E:/cits3001p/models/best_model_200000")

observation = env.reset()

# Run agent in environment
while True:
    action, _ = model.predict(observation)
    state, reward, done, info = env.step(action)
    env.render()