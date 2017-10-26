# Homework1 report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.

## Overview
In this project, we try to go throught Markov Decision Procession (MDPs). In following practices, we will try to implement value iteration, policy iteration and tabular Q-learning.
To make this project more interesting, the designer tried to set some senario to make problems close to our real life.
In Problem 1 and Problem 2, we are solving the problem that trying to get frisbee on iced lake and preventing falling into holes. In these two part, we used FrozenLakeEnv() environment.
In Problem 3, we try to train a robot to walk faster. In thos project, we used CrawlingRobotEnv() environment.
## Problem 1 : implement value iteration
In this part, I will calculate all values in Q. Then fill the largest value of Q to V and log its arg in pi. For implementation, I used three loops one for all possible states, one for four actions and the other for all probability in mdp.p.

## Problem 2 : Policy Iteration
### a. state value function
In this part, I follow the hint in the description of problem to calculate a and b. Bacause all policy is defined, so we don't need to go throught all actions. Then, I used np.linalg.solve(a, b) to calculate V value. 

### b. state-action value function
This part, we need to calculate state value function. I used three for loop to implement this equation. The detail of implementation is pretty similar to problem one. Then all functions we need for policy iteration are ready. We just need to call compute_vpi and compute_qpi respectively, we can get the result.


## Problem 3 : Sampling-based Tabular Q-Learning
First, we need to prepare some function required when iteration. One is eps_greedy. This function would help us to find a random action with prbability of epsilon and get index of Q which has largest value with probability of (1-epsilon). The other function is q_learning_update. This function helps us to update our Q table value once we take action. Then we can start our iteration by calling these function respectively. 
