import gymnasium as gym
#import gym
from stable_baselines3 import DQN

training_steps = 3000000

env_id="CartPole-v1"
env = gym.make(env_id, render_mode="rgb_array")
model = DQN("MlpPolicy", "CartPole-v1",
            learning_rate=0.0001, exploration_fraction=0.2,
            verbose=1, tensorboard_log="logs/dqn_cartpole_tensorboard/")
model.learn(total_timesteps=training_steps, log_interval=3)
model.save(f"models/cartpole_guide_{training_steps}")