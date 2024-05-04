# Arctic Adventure, Q-learning for Frozen Lake

<p>
<img alt="NumPy" src="https://img.shields.io/badge/-NumPy-blue?style=flat-square&logo=NumPy&logoColor=white" />
<img alt="python" src="https://img.shields.io/badge/-Python-13aa52?style=flat-square&logo=python&logoColor=white" />
</p>

This repository contains a Python implementation of Q-Learning, a popular reinforcement learning algorithm, designed to learn optimal action-selection policies for Markov decision processes.

## Overview

Q-Learning is a model-free, off-policy reinforcement learning algorithm used to find the optimal action-selection policy for a given finite Markov decision process. It does not require a model of the environment and is capable of learning optimal policies directly from interactions with the environment.

This implementation includes a Q-learning class (*Q_learning*) that can be utilized to train agents in various environments and tasks.

## Results

![Frozen Lake](figs/frozen_lake.gif)

The Q-learning algorithm was tested on the Frozen Lake environment, a popular grid-world environment in OpenAI Gym. The agent was able to learn an optimal policy for navigating the frozen lake and reaching the goal state.

![Rewards for Q-Learning](figs/frozen_lake_rewards.png)

## Usage

Please do not use this code for your homework assignments.
