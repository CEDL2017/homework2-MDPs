# 賴承薰 <span style="color:red">(105061583)</span>

## HW 2: MDPs

## Overview
The assignment is related to:
Markov Decision Process: value iteration, policy iteration, tabular Q-learning

## Envs
* python 3.5
* OpenAI gym
* Numpy
* Matplotlib
* iPython (Jupyter)

## Implementation
### 1. Value iteration

```
 V = np.zeros(mdp.nS)
        pi = np.zeros(mdp.nS)
        for s in range(mdp.nS):
            bellman = np.zeros(mdp.nA)
            for a in range(mdp.nA):
                for prob, next_state, reward in mdp.P[s][a]:
                    bellman[a] += prob * (reward + gamma * Vprev[next_state])
            V[s] = np.max(bellman)
            pi[s] = np.argmax(bellman)
```

### 2. Policy iteration

1) compute state value function Vpi
```
A = np.eye(mdp.nS)
  b = np.zeros(mdp.nS)
  V = np.zeros(mdp.nS)

  for s in range(mdp.nS):
    for prob, next_state, reward in mdp.P[s][pi[s]]:
      A[s][next_state] -= gamma * prob
      b[s] += prob * reward
  V = np.linalg.solve(A, b)
```

2) compute state-action value function Qpi
```
Qpi = np.zeros([mdp.nS, mdp.nA]) # REPLACE THIS LINE WITH YOUR CODE
for s in range(mdp.nS):
    for a in range(mdp.nA):
        for prob, next_state, reward in mdp.P[s][a]:
            Qpi[s][a] += prob * (reward + gamma * vpi[next_state])
```
3) policy iteration
```
vpi = compute_vpi(pi_prev, mdp, gamma)
pi = compute_qpi(vpi, mdp, gamma)
pi = np.argmax(pi, axis = 1)
```

### 3. Tabular Q-learning

## Installation
* Use the tool Jupyter and make sure choosing the kernel Python3
* `shift + enter` to run the section, error message would be reported rught below the box
* Ensure Matplotlib, Numpy, gym are installed before running


## Results

problem 1: value_iteration

<img src="螢幕快照 2017-10-17 下午6.55.12.png" width="50%"/>

------------------------------------------------------------------------------------------------

problem2a: state value function

<img src="螢幕快照 2017-10-17 下午6.54.51.png" width="70%"/>

------------------------------------------------------------------------------------------------

problem2b: state-action value function

<img src="螢幕快照 2017-10-17 下午6.54.25.png" width="50%"/>

------------------------------------------------------------------------------------------------

Problem 3: Sampling-based Tabular Q-Learning
<video autoplay>
    <source src="crawler.mov" type='video/mov'/>
</video>
