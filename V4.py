#import gym
import gymnasium as gym # Import gymnasium instead of gym
import numpy as np
from stable_baselines3 import PPO
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy.integrate import quad#import gym
import gymnasium as gym # Import gymnasium instead of gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
#from exoskeleton_env import ExoskeletonEnv
from V3 import *

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the environment and wrap it for monitoring
    env = ExoskeletonEnv()
    env = Monitor(env)

    # Evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # Create the PPO agent with custom hyperparameters learning_rate=3e-4,
    ppo_model = PPO(
        "MlpPolicy",
        env,
        learning_rate= 3e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device=device
    )


    # Train the agent with the callback
    ppo_model.learn(total_timesteps=1000000, callback=eval_callback)
    print("### Leaning done")

    # Save the trained model
    ppo_model.save("test")
    print("###  Model Saved")

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(ppo_model, env, n_eval_episodes=7)
    # Test the agent and plot results
    obs, _ = env.reset()
    rewards = []
    for _ in range(5000):
        action, _states = ppo_model.predict(obs)
        if env.needs_reset:
            env.reset()
        obs, reward, done, info, _ = env.step(action)
        rewards.append(reward)
        if done:
            obs = env.reset()

    # Plot the reward over time
    plt.plot(rewards)
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title("PPO - Rewards Over Time_test_100lstm")
    plt.savefig("step_rewards_test_100lstm.svg")
    plt.savefig("step_rewards_test_100lstm.png")
    plt.show()

    # Plot learning curve (total reward per episode)
    results = env.get_episode_rewards()
    plt.plot(results)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Training Progress_test_100lstm')
    plt.savefig("episodes_rewards_test_100lstm.svg")
    plt.savefig("episodes_rewards_test_100lstm.png")
    plt.show()
        # Plot the reward over time
