# 江愷笙 <span style="color:red">(106062568)</span>

# Homework1 report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.

## Overview
This homework is related to UC Berkerly cs294 MDPs.

> Markov Decision Process is a discrete time stochastic control process. At each time step, the process is in some state s, and the decision maker may choose any action a that is available in state s. The process responds at the next time step by randomly moving into a new state s', and giving the decision maker a corresponding reward R(s,a,s')
## Implementation
In this homework we have to implement value iteration, policy iteration and tabular Q-learning. In value and policy iteration, we use a grid world which simulate a frozen lake with goal and hole on it. For an action on a specific state, mdp.P[state][action] will return possible next state and the probability and reward accordingly.
* Value Iteration

For value iteration, we have to implement two math equation below.
<tr>
<td>
<img src="imgs/value_1.png"/>
<img src="imgs/value_2.png"/>
</td>
</tr>
```python
V = np.zeros(mdp.nS)
pi = np.zeros(mdp.nS)
for s in range(mdp.nS):
    V_act = np.zeros(mdp.nA)
    for a in range(mdp.nA):
        for prob, s_, reward in (mdp.P[s][a]):
            V_act[a] = V_act[a] + prob * (reward + gamma * Vprev[s_])
    V[s] = np.amax(V_act)
    pi[s] = np.argmax(V_act)
```

## Installation
* Anaconda
* Ipython notebook
* Python3.5
* OpenAI gym
## Results


