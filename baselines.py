'''
This agent has taken inspiration from Nicholas Renotte's guide titled 
'Build an Mario AI Model with Python'. The full video supplied was watched
and used as a guideline and inspiration to using Stable Baselines

This agent is not hand made or originally made using Stable Baselines. Rather,
this model is simply used as a third point of comparison for our 
original models.

Please note the reused code is available a the following GitHub link:
https://github.com/nicknochnack/MarioRL

Comments starting with ## are personal changes to assist
#With our understanding

'''
##Importing dependencies for PPO + StableBaslines

# Import os for file path management
import os 
# Import PPO for algos
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np
import string
#Please not we are only using the simple movements

##A possible extension of this model is to use the COMPLEX_MOVEMENT


#Setup game
env = gym_super_mario_bros.make('SuperMarioBros-v0',apply_api_compatibility=True,render_mode="human" )
env = JoypadSpace(env, SIMPLE_MOVEMENT)


# Create a flag - restart or not
done = True
# Loop through each frame in the game
for step in range(10): 
    # Start the game to begin with 
    if done: 
        # Start the game
        env.reset()
    # Do random actions
    ##Here we had to add ,_ in order to ignore the additional values
    ##Returned from doing a random action
    state, reward, done, info,_ = env.step(env.action_space.sample())
    ##env.action_space.sample() is what does the random actions
    # Show the game on the screen
    env.render()
# Close the game
env.close()


###############PREPROCESSING########################
##Installed both packages 


# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt

# 1. Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 2. Simplify the controls 
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
#4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')
## 4 frames to stack
## 

'''
state, reward, done, info = env.step([5])

plt.figure(figsize=(20,16))
for idx in range(state.shape[3]):
    plt.subplot(1,4,idx+1)
    plt.imshow(state[0][:,:,idx])
plt.show()
'''
## Using PPO 
##Importing dependencies for PPO + StableBaslines

##Saves the model (callback funtion to save the model x number of steps)
## check_freq = how many steps until it saves
## save_path = where the model will be saved

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
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
    

##Creating directories for where we will store the trained models
##inside of /logs, we create a new log file. We get a tensor flow log file
##Which enables us to use tensor board to see a range of visualisations

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

# Setup model saving callback

########################NOTE##########################
##This will save a new model every 100000 steps. Each save is ~300mb so 
##Please be careful and adjust based on how much disk space you have

callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)
 
# This is the AI model started

## We are using CnnPolicy as it is very strong at processing images
##Tensorboard log helps us see performance metrics
##Learning rate - can adjust for different results

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, 
            n_steps=512) 

# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=1000000, callback=callback)


###################TESTING THE MODELS###################
# Load model
model = PPO.load('./train/best_model_1000000')


# Start the game 
state = env.reset()
# Loop through the game
while True: 
    ##Using our trained model to get our next action
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()