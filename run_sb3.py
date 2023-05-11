import gymnasium as gym
#import gym
from stable_baselines3 import JSRLDQN, DQN
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from custom_callbacks import EvalJsrlCallback, TensorboardCallback

env_id="CartPole-v1"
env = gym.make(env_id, render_mode="rgb_array")

guide_policy = DQN.load("models/cartpole_guide_3000000_best")

for schedule in ["random", "successive"]:
    for beta in [0, 100, 200, 300]:
        for i in range(5):
            path = f"jsrldqn_cartpole_beta{beta}_{schedule}"
            policy_kwargs = dict(guide_policy=guide_policy,
                                max_horizon=500,
                                horizon_schedule=schedule)
            model = JSRLDQN("JsrlPolicy",
                            "CartPole-v1",
                            policy_kwargs=policy_kwargs,
                            learning_rate=0.0001,
                            learning_starts=0,
                            exploration_fraction=0.2,
                            verbose=0,
                            tensorboard_log=f"./logs/{path}/")
            # Separate evaluation env
            eval_env = gym.make(env_id)
            # Use deterministic actions for evaluation
            eval_callback = EvalJsrlCallback(eval_env, 
                                             best_model_save_path=f"./models/{path}_{i}",
                                             log_path=f"./logs/{path}",
                                             eval_freq=1000,
                                             deterministic=True,
                                             render=False,
                                             reward_threshold=beta)
            tb_callback = TensorboardCallback()

            model.learn(total_timesteps=1000000, log_interval=10, callback=[eval_callback, tb_callback])
