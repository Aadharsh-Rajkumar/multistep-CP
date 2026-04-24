# train_policy.py
#
# This file originally used OpenAI baselines (PPO) to train a locomotion
# policy for the Hopper-v2 MuJoCo environment. The trained policy was then
# used by the hopper notebook to collect trajectory data for ALPaCA.
#
# OpenAI baselines requires TensorFlow 1.x and is not compatible with
# Python 3.13+. If you need to train a new hopper policy, you have two options:
#
# Option 1: Use stable-baselines3 (pip install stable-baselines3)
#   from stable_baselines3 import PPO
#   model = PPO("MlpPolicy", "Hopper-v4", verbose=1)
#   model.learn(total_timesteps=1_000_000)
#   model.save("policies/hopper")
#
# Option 2: If you already have trajectory data saved as .npz files,
#   you can skip policy training entirely and load the data directly
#   in the hopper notebook.
#
# The ALPaCA model training (the core of this repo) does NOT depend on
# this file — it only needs trajectory data in numpy array format.

import os
import warnings


def train(*args, **kwargs):
    warnings.warn(
        "train_policy.py originally used OpenAI baselines (TF 1.x). "
        "Consider using stable-baselines3 for policy training on Python 3.13+. "
        "See comments at the top of this file for instructions.",
        DeprecationWarning,
        stacklevel=2
    )
    raise NotImplementedError(
        "train_policy.py requires OpenAI baselines which is not compatible "
        "with Python 3.13+. See this file's comments for alternatives."
    )


def main():
    train()


if __name__ == '__main__':
    main()
