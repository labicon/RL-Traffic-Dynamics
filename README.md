# RL-Traffic-Dynamics

[![License:
 MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 


## Overview
This is the public code repo for the paper at ICRA 2023 London , [Learning to Influence Vehicles' Routing in Mixed-Autonomy Networks by Dynamically Controlling the Headway of Autonomous Cars](https://arxiv.org/abs/2303.04266), by Xiaoyu Ma and Prof. Negar Mehr from University of Illinois at Urbana-Champaign.

![](gitrepo_imgs/dgm.PNG)

In this work, we propose that in mixed-autonomy networks, i.e., networks where roads are shared between human-driven and autonomous cars, the headway of autonomous cars can be directly controlled to influence vehicles' routing and reduce congestion. We argue that the headway of autonomous cars -- and consequently the capacity of link segments -- is not just a fixed design choice; but rather, it can be leveraged as an {infrastructure control} strategy to {dynamically} regulate capacities. 

This code repo provided is developed as a custom environment on top of [OpenAI Gym](https://openai.com/research/openai-gym-beta). By defining the classic traffic network -- Braess network in the format of a gym env, we employed [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) to harness the power of Reinforcement Learning. The current implementation provides an one-on-one control scheme for each link in the classic Braess network. By learning a policy to generate varying headway for links, the resulted vehicle distribution across the network is very different from the one without controlling (constant headway), as shown below, where red links are more crowded and green links are less congested.

Constant Headway (without control)          |  Varying Headway (with control)
:-------------------------:|:-------------------------:
![](/gitrepo_imgs/braess_constant_headway.gif)  |  ![](/gitrepo_imgs/braess_varying_headway.gif)


The repository include necessary files for running the RL_headway_dynamics project in the classic Braess network. Both Linex local implmentation and jupyter notebook implementation are included for demonstration purpose.

## Local Implementation

To use the code implementation locally, you need to setup a Linex system and install python3+. After that, use ```pip``` to install required packages before moving on to register our gym env. 

### Required packages
torch, gym, stable_baselines3, wandb

```gym-traffic/gym_traffic/envs``` contains the defined traffic model

Please register the env file as a self-defined env in gym to utilize the code if only using the .py files for executing before running the runner.py


## Jupyter Notebook Implementation
Without registering the env, you can run the notebook in any Linux environment or online server interface (Google CoLab for instance).
