# Conservative Q-Learning (CQL) Implementation

This repository contains an implementation of Conservative Q-Learning for offline reinforcement learning. The implementation was tested on Half-Cheetah and InvertedPendulum environments. However, due to issues with offline RL datasets 

## Steps
You can either run `python main.py` or run the Jupyter Notebook on either your device or in Google Colab. The second method is the easiest and the recommended method for reproducing the results.

## Explanation of Files
- ``main.py``: Main file for initiallizing the Gym environment, completing an online RL training loop, and logging training performance
- ``model.py``: CQL Implementation
- ``replay_buffer.py``: Replay buffer for storing [state, action, reward, next_state, done] tuples
- ``requirements.txt``: List of dependencies

## Necessary Dependencies
This program was run and tested in Python 3.8+. Please ensure your Python version is 3.8+.
Additionally, some Python libraries used are:
- torch
- gym

You can install the dependencies using `pip install -r requirements.txt`. Using the Jupyter Notebook will install everything already for you.