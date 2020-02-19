[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Project 2 - Udacity DRL: Continuos Control

## Project Details

For this project, I will train an agent to solve the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment, that was developed by [Unity]( https://unity3d.com/es).

### Environment Description
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The version used in this project uses 20 replicas of the arm and to be considered resolved it must meet the following criteria:

- In particular, agents must obtain an average score of +30 (more than 100 consecutive episodes and over all agents). Specifically,
- After each episode, the rewards that each agent receives are added and then the average of these 20 scores is taken.
- This produces an ** average score ** for each episode (where the average is over all 20 agents).

The environment is considered resolved when the average (more than 100 episodes) of these average scores is at least +30.

## Getting Started

For this project, you are going to need some basic libraries like _numpy_ and _matplotlib_. For the neural network of the agent, we are going to use PyTorch. In order to be able to interact with the Unity Toolkit we have to install the following version of torch and torchvision.

* __torch=0.4.0__
* __torchvision=0.2.1__

You can see instructions for installation these old libraries directly on the PyTorch [website](https://pytorch.org/get-started/previous-versions/)

### Download the environment

1. You can download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
2. Place the file in the `root/` folder of the project, and unzip (or decompress) the file.


## Instructions

This repository contains the following scripts
- __dqn_agent.py__
    - This script manages the internal variables of the agent as well as the training phase. 
    - The Actor and the Critic models are located here and all the interaction with the environment. 
    - It also manages the replay memory buffer.
- __model.py__
    - Here is where the architectures of the models are initialized.
- __environment.py__
    - It is a simplified script to interact with the Unity environment. It has two main methods:
        - __reset:__ resets the environment and returns the state of the env.
        - __execute:__ receives an action and returns the next state, the reward and a bool value indicating if the env is in a termnal state.
- __train_DDPG_reacher.ipynb__
    - This script allows you to see step by step the training phase.

Open the __train_DDPG_reacher.ipynb__ and execute the cells to see how to interact with the environment. After you will have the opportunity to see an untrained agent acting. Finally, you will train the agent.