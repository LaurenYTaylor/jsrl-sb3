import gymnasium as gym
#import gym
from stable_baselines3 import DQN


env_id="CartPole-v1"
env = gym.make(env_id, render_mode="rgb_array")
model = DQN("MlpPolicy", "CartPole-v1", learning_rate=0.0005, verbose=1,
            tensorboard_log="./dqn_cartpole_tensorboard/")
model.learn(total_timesteps=300000, log_interval=3)
model.save("models/cartpole_guide")