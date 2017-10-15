# 106061532洪彗庭 CEDL-hw2

In this assignment we have implemented value and policy iteration on MDP's dynamics model and also use sampling-based Q-Learning to learn to control a Crawler robot.

## Setup
Simply follow the instructions from TA, like follows:

```
conda env create -f environment.yml
source activate cedl
# deactivate when you want to leave the environment
source deactivate cedl
```

## Implementation
### Simple Gridworld - Frozen Lake environment
env </br>
state transition and reward probability </br>
here it defines a 2-level dictionary, **mdp.P**, to represent the states, possible actions in each state, and their rewards.
#### Value iteration

#### Policy iteration
### Sampling-based Tabular Q-Learning