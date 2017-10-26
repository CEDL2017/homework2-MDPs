# Homework1 report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.
## Problem 1: implement value iteration
<p>First, I'm required to implement value iteration(value update, Bellman update/back-up) which was mentioned in the class. </p>
<img src='./imgs/problem1.png' width = '50%' height = '50%'>
<p>Here's my implementation</p>
<img src='./imgs/problem1_code.png' width='50%' height = '50%'>
## Problem 2: Policy Iteration
### Problem 2a: state value function
<p>Here, we're going to the exact value function. First find a(idendity matrix - gamma*P) and b(sum of P*R), then use numpy.linalg.solve to find the value function V</p>
<img src='./imgs/problem2_1_code.png' width='50%' height = '50%'>
### Problem 2b: state-action value function
<p>Since we have the state value function, then we can calculate state-action value function which is denoted as Qpi</p>
<img src='./imgs/problem2_2_code.png' width='50%' height = '50%'>
## Problem 3: Sampling-based Tabular Q-Learning
<p>If the environment is given as a blackbox physics simulator, then we won't be able to read off the whole transition model.So in this problem we're going to solve using sampling-based tabular Q-learning.</p>
<p>First, we randomly explore(select actions) the environmen5t or take some action based on existing experiences(greedy)</p>
<img src='./imgs/problem3_code.png' width='50%' height = '50%'>
<p>Then implement the Q-balue update function</p>
<img src='./imgs/problem3_code_2.png' width='50%' height = '50%'>
<p>Finally, we can combine them together to complete the agent. For each iteration, we make action, get the reward from environment, update Q-function, update current state, ... next iteration</p>
<img src='./imgs/problem3_code_3.png' width='50%' height = '50%'>