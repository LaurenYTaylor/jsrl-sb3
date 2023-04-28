import gymnasium as gym
#import gym
from stable_baselines3 import JSRLDQN, DQN
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

env_id="CartPole-v1"
env = gym.make(env_id, render_mode="rgb_array")

guide_policy = DQN.load("models/cartpole_guide")

policy_kwargs = dict(guide_policy=guide_policy,
                     max_horizon=500,
                     horizon_schedule="successive")

model = JSRLDQN("JsrlPolicy",
            "CartPole-v1",
            policy_kwargs=policy_kwargs,
            learning_rate=0.0005,
            learning_starts=0,
            verbose=1,
            tensorboard_log="./jsrldqn_cartpole_tensorboard/")

'''
# Separate evaluation env
eval_env = gym.make("Pendulum-v1")
# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=500,
                             deterministic=True, render=False)
'''

model.learn(total_timesteps=300000, log_interval=3)

'''
obs = vec_env.reset()
for i in range(100):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render(mode="human")
    # VecEnv resets automatically
    #if done:
     #  obs = eval_env.reset()
'''