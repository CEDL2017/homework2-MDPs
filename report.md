# 江愷笙 <span style="color:red">(106062568)</span>

# Homework1 report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.

## Overview
This homework is related to UC Berkerly cs294 MDPs.

> Markov Decision Process is a discrete time stochastic control process. At each time step, the process is in some state s, and the decision maker may choose any action a that is available in state s. The process responds at the next time step by randomly moving into a new state s', and giving the decision maker a corresponding reward R(s,a,s')
## Implementation
In this homework we have to implement value iteration, policy iteration and tabular Q-learning. In value and policy iteration, we use a grid world which simulate a frozen lake with goal and hole on it. For an action on a specific state, mdp.P[state][action] will return all possible next state and the probability and reward accordingly.
* Value Iteration

For value iteration, we have to implement two math equation below.

<table border=1>
<tr>
<td>
<img src="imgs/value_1.PNG"/>
<img src="imgs/value_2.PNG"/>
</td>
</tr>
</table>
To update the value function for a given state, we have to go through each action in the state. And for each action, we have to sum over all the possible next state and finally, figure out the max value of the actions and update it to the value function in this state, and update the index of the action to the policy of the state.

```python
V = np.zeros(mdp.nS)
pi = np.zeros(mdp.nS)
for s in range(mdp.nS):
    V_act = np.zeros(mdp.nA)
    for a in range(mdp.nA):
        for prob, s_, reward in (mdp.P[s][a]):
            V_act[a] = V_act[a] + prob * (reward + gamma * Vprev[s_])	# sum over all the possible next state
    V[s] = np.amax(V_act)						# update the max value of the action to the value function
    pi[s] = np.argmax(V_act)						# update the index of this action to the policy
```

* Policy Iteration

For policy iteration we have to compute state value function first and then the state action value function, finally, combine them to complete the policy iteration.

<table border=2>
<tr>
<td>
<img src="imgs/policy_1.PNG"/>
<img src="imgs/policy_2.PNG"/>
</td>
</tr>
</table>
We use the matrix form to solve the linear equation to obtain state value function. (I - gamma * P)V = P * R, which is equal to the matrix form Ax = B.

```python
a = np.zeros((mdp.nS, mdp.nS)) 
b = np.zeros(mdp.nS)
I = np.identity(mdp.nS)				# generate an identity matrix
P = np.zeros((mdp.nS, mdp.nS))
for s in range(mdp.nS):
    for prob, s_, reward in mdp.P[s][pi[s]]:	# for a given policy in the state
        P[s][s_] += prob
        b[s] += prob * reward			# here we cannot use another array R to sum over all the reward
a = I - (gamma * P)				# and then do the matrix multiplication P*R
V = np.linalg.solve(a, b)			# since each reward has its corresponding probability
```

State action value function is a 2D array which records the value for each state and each action accordingly.

```python
Qpi = np.zeros([mdp.nS, mdp.nA])
for s in range(mdp.nS):
    for a in range(mdp.nA):
        for prob, s_, reward in mdp.P[s][a]:
            Qpi[s][a] += prob * (reward + gamma * vpi[s_])
```

Finally we combine the two function to do the policy iteration.

```python
for it in range(nIt):
    vpi = compute_vpi(pi_prev, mdp, gamma)
    qpi = compute_qpi(vpi, mdp, gamma)
    pi = np.argmax(qpi,1)	# the policy is the index of the max value for each state 
    				# (i.e. the action with the max value)
```

* Tabular Q-learning

For tabular Q-learning we use another environment for the crawling robot.
To implement tabular Q-learning, we have to introduce greedy epsilon, which gives the random action to the robot. For a given epsilon, the probability to random select an action is epsilon, and the probability to select the action with max value is (1 - epsilon). The reason to include the randomness is that the world for the robot is not totally equal to our real world. For example, we ask the robot to move right, it still have some probability for the robot to do other actions such as moving left. Another reason is that if we add some randomness, the robot may not act too conservative, and this can avoid the robot to stuck in the local minima. For example, if some action leads to the negtive reward, it is likely that the robot would stuck in current state in order not to decrease the value, however, it is hard to achieve the goal. So it is helpful to add some randomness into it.

For epslon greedy, if the random number smaller than epsilon, then random choose the action, otherwise, choose the action with the max value.

```python
import random
action = 0
random = random.random()
if random < eps:
    action = np.random.choice(range(len(q_vals[state])))	# random choose the action
else:
    action = np.argmax(q_vals[state])				# choose the action with the max value
```
To update the Q table, we have to implement the following two mathematical equations

<table border=3>
<tr>
<td>
<img src="imgs/Q-learning_1.PNG"/>
<img src="imgs/Q-learning_2.PNG"/>
</td>
</tr>
</table>

```python
target = reward + gamma * np.amax(q_vals[next_state])
q_vals[cur_state][action] = (1-alpha) * q_vals[cur_state][action] + alpha * target
```

Finally we combine these together

```python
for itr in range(300000):
    action = eps_greedy(q_vals, eps, cur_state)
    next_state, reward, done, info = env.step(action)		# obtain next_state and reward from the environment
    q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward)
    cur_state = next_state
```

## Installation
* Anaconda
* Ipython notebook
* Python3.5
* OpenAI gym
* to run the code, open Lab2-MDPs.ipynb by using Ipython notebook and execute each block.

## Results
* Value Iteration

Here we can see the action and value update for each state.

<table border=4>
<tr>
<td>
<img src="imgs/VI_1.PNG" width="19%"/>
<img src="imgs/VI_2.PNG" width="19%"/>
<img src="imgs/VI_3.PNG" width="19%"/>
<img src="imgs/VI_4.PNG" width="19%"/>
<img src="imgs/VI_5.PNG" width="19%"/>
<img src="imgs/VI_6.PNG" width="19%"/>
<img src="imgs/VI_7.PNG" width="19%"/>
<img src="imgs/VI_8.PNG" width="19%"/>
<img src="imgs/VI_9.PNG" width="19%"/>
<img src="imgs/VI_10.PNG" width="19%"/>
<img src="imgs/VI_plot.PNG"/>
</td>
</tr>
</table>

* Policy Iteration

<table border=5>
<tr>
<td>
<img src="imgs/PI_plot.PNG"/>
</td>
</tr>
</table>

* Tabular Q-learning

By finishing the tabular Q-learning, we can see a crawling robot moving fast from the left to the right.

<table border=5>
<tr>
<td>
<img src="imgs/Q_1.PNG"/>
<img src="imgs/Q_2.PNG"/>
</td>
</tr>
</table>
