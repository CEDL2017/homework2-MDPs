# Homework2-MDPs 105061535

## Value Iteration

```python

V = np.zeros(mdp.nS)
pi = np.zeros(mdp.nS)
for i in range(mdp.nS):
    Vtmp = np.zeros(mdp.nA)
    for j in range(mdp.nA):
        for k in mdp.P[i][j]:
            Vtmp[j] += k[0] * (k[2] + gamma * Vprev[k[1]])
    V[i] = max(Vtmp)
    pi[i] = np.argmax(Vtmp)
```
先initial value function 和 policy。接著for loop所有state, action，算出會得到的reward，最後挑reward最大的當作這個state的value，policy則挑可以得到最大reward的action。


## Policy Iteration

### State Value Function

```python
a = np.eye(mdp.nS) 
b = np.zeros(mdp.nS) 
    
for s in range(mdp.nS):
    for (p, next_s, r) in mdp.P[s][pi[s]]:
        a[s][next_s] -= gamma * p
        b[s] += p * r
    
V = np.linalg.solve(a, b)
```
這邊要evaluate這次policy的value function。這可以當作一個linear system解出來，把公式整理一下，得到a, b就可以算出。
for loop所有state和next state算出a, b就可以解出V。


### State-Action Value Function

```python
Qpi = np.zeros([mdp.nS, mdp.nA]) # REPLACE THIS LINE WITH YOUR CODE
    
for s in range(mdp.nS):
    for a in range(mdp.nA):
        transition_info = mdp.P[s][a]
        for (p, next_s, r) in transition_info:
            Qpi[s][a] += p * (r + gamma * vpi[next_s])
```

根據上面算好的Vpi，可以算出Qpi，只要for loop所有state, action，用Q function公式就可得到。


### policy_iteration

```python
for it in range(nIt):
    # you need to compute qpi which is the state-action values for current pi
    #               and compute the greedily policy, pi, from qpi
    # >>>>> Your code (sample code are 3 lines)
    vpi = compute_vpi(pi_prev, mdp, gamma)
    qpi = compute_qpi(vpi, mdp, gamma)
    pi = np.argmax(qpi, 1)
```
Policy iteration的作法事先用現在的policy去算出Vpi，再用Vpi去update policy。
用上面寫好的function，得到Vpi, Qpi最後update pi。
## Problem 3: Sampling-based Tabular Q-Learning

```python
import random
random_num = random.random()
if random_num < eps:
    action = random.randint(0, len(q_vals[state])-1)
else:
    action = np.argmax(q_vals[state])
```
在sampling-based情況下要從探索或是使用現在的policy去作曲捨，可以用eps來當作使否探索的機率，如果機率小於eps則隨機選一個動作探索，否則就用現有的policy做決定。


```python
target = reward + gamma * np.max(q_vals[next_state])
q_vals[cur_state][action] = (1-alpha) * q_vals[cur_state][action] + alpha * target
```
update q value得作法。



```python
for itr in range(300000):
    # YOUR CODE HERE
    # Hint: use eps_greedy & q_learning_update
    # >>>>> Your code (sample code are 4 lines)
    
    action = eps_greedy(q_vals, eps, cur_state)
    next_state, reward, done, info = env.step(action)
    q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward)
    cur_state = next_state
```
先用eps_greedy選擇action，環境給我們reward, next_state，在update q value




