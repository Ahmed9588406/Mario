# Import the game
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time
import gym

# Setup game

# this is an environment of the game with a certin pixels to show i  found this the best one of them all :)
env = gym_super_mario_bros.make('SuperMarioBros-v0')


# that makes the movement from 256 that hard for AI model to learn and train to a 7 actions

env = JoypadSpace(env, SIMPLE_MOVEMENT)

"""
# Create a flag - restart or not
done = True
# Loop through each frame in the game
for step in range(100000):
    # Start the game to begin with
    if done:
        # Start the game
        env.reset()
    # Do random actions
    state, reward, done, info = env.step(env.action_space.sample())
    # Show the game on the screen
    env.render()
# Close the game
env.close()
"""
# Import Frame Stacker Wrapper and GrayScaling Wrapper

# FrameStack : used to stack a frames say four frames to make ai model to know what is a good action and not a good one from the several frames that stacked
# GrayScaleObservation : make us deal with an gray photo rather than a photo with 3 colors that make our model perform well
from gym.wrappers import FrameStack, GrayScaleObservation

# Import Vectorization Wrappers
# VecFrameStack : to make us deal with FrameStack
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt


# 1. Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')

# 2. Simplify the controls


# 3. Graystyle
env = GrayScaleObservation(env, keep_dim=True)

# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])

# 5. Stack the frames
# channels_order : this make the last order as i give it 4 ex: state = env.reset() state = env.reset()
# the output will be (1,240,256,4) 4 is the last order and the number of frames the we will stack an make the model learn on
env = VecFrameStack(env, 4, channels_order='last')# this wil be a good number of frames to train

"""
state = env.reset()

state, reward, done, info = env.step([5])

plt.figure(figsize=(20, 16))
for idx in range(state.shape[3]):
    plt.subplot(1, 4, idx+1)
    plt.imshow(state[0][:, :, idx])
plt.show()
"""

# Train the RL Model

# Import os for file path management

import os

# Import PPO Algorithm
from stable_baselines3 import PPO

# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback



# allow us to save our model  after x number of steps
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


# i make this to save the models here
Checkpoint_dir = './train/'


# Here every time i run the algorithm we creat a file here in logs to see number of times i run the PPO Algorithm
# in every PPO file there is a tensorflow log
LOG_DIR = './logs/'


# Setup model saving callback
# every 10000 step new model is saved
callback = TrainAndLoggingCallback(check_freq=10000, save_path=Checkpoint_dir, verbose=1)

# this is the AI model Started
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)


"""
# Train the AI model , this is where the AI model starts to learn
model.learn(total_timesteps=1000000, callback=callback)
"""




# Load model
model = PPO.load('./train/best_model_70000.zip')

state = env.reset()
# Start the game
state = env.reset()
# Loop through the game
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()





