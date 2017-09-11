# Homework1-MDPs

The lab materials are partially borrowed from [UC Berkerly cs294](http://rll.berkeley.edu/deeprlcourse/)


## Introduction
In this homework, we solve MDPs with finte state and action space via value iteration, policy iteration, and tabular Q-learning. 

### What's Markov decision process (MDP)?
Markov Decision Process is a discrete time stochastic control process. At each time step, the process is in some state `s`, and the decision maker may choose any action `a` that is available in state `s`. The process responds at the next time step by randomly moving into a new state `s'`, and giving the decision maker a corresponding reward `R(s,a,s')`


<p align="center"><img src="imgs/mdps.png" height="300"/></p>

image borrowed from UCB CS188

## Setup
- Python 3.5.3
- OpenAI gym
- numpy
- matplotlib
- ipython
All the codes you need to modified are in ```Lab2-MDPs.ipynb```. 

We encourage you to install [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://conda.io/miniconda.html) in your laptop to avoid tedious dependencies problem.

**for lazy people:**
```
conda env create -f environment.yml
source activate cedl
# deactivate when you want to leaving the environment
source deactivate cedl
```

## Prerequisites

If you are unfamiliar with Numpy or IPython, you should read materials from [CS231n](http://cs231n.github.io/):

- [Numpy tutorial](http://cs231n.github.io/python-numpy-tutorial/)
- [IPython tutorial](http://cs231n.github.io/ipython-tutorial/)


## How to Start

Start IPython: After you clone this repository and install all the dependencies, you should start the IPython notebook server from the home directory
Open the assignment: Open Lab1-MDPs (students).ipynb, and it will walk you through completing the assignment.

## TODO
- [30%] value iteration
- [30%] policy iteration
- [30%] tabular Q-learning
- [10%] report

# Other
If you stuck in the homework, here are some nice material that you can take it a look :smile:
- [dennybritz/reinforcement-learning](https://github.com/dennybritz/reinforcement-learning)
- [UC Berkeley CS188 Intro to AI](http://ai.berkeley.edu/home.html)
