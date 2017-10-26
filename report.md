# Homework2 Report 105061525

## Problem 1: implement value iteration

```python
V = np.zeros(mdp.nS)
pi = np.zeros(mdp.nS)
        
for s in range(mdp.nS):
    V_s_tmp = np.zeros(mdp.nA)
    for a in range(mdp.nA):
        transition_info = mdp.P[s][a]
        for (p, next_s, r) in transition_info:
            V_s_tmp[a] += p * (r + gamma * Vprev[next_s])
    pi[s] = np.argmax(V_s_tmp)
    V[s] = V_s_tmp[int(pi[s])]
```

這裡我用`V_s_tmp`儲存在state s時，執行所有action會得到的value。其中value的算法是這個action得到的reward加上上一個iteration的value function在next state的value。
接著選擇value最大的action並update value。


## Problem 2: Policy Iteration

### Problem 2a: state value function

```python
a = np.eye(mdp.nS) 
b = np.zeros(mdp.nS) 
    
for s in range(mdp.nS):
    for (p, next_s, r) in mdp.P[s][pi[s]]:
        a[s][next_s] -= gamma * p
        b[s] += p * r
    
V = np.linalg.solve(a, b)
```

state value function是一個linear的function，所以可以用`np.linalg.solve`去解。把function整理成這個形式: (I-gamma * P) * V = P * R。
a就是(I-gamma * P)，b就是P * R。


### Problem 2b: state-action value function

```python
Qpi = np.zeros([mdp.nS, mdp.nA]) # REPLACE THIS LINE WITH YOUR CODE
    
for s in range(mdp.nS):
    for a in range(mdp.nA):
        transition_info = mdp.P[s][a]
        for (p, next_s, r) in transition_info:
            Qpi[s][a] += p * (r + gamma * vpi[next_s])
```

state-action value function 要計算現在這個state執行每個action所得到的value。
而value是來自現在這個policy得到的所有state的value function。


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

最後對計算出來的Q function取argmax就可以得到每個state的最佳action，也就是下次iteration新的policy。


## Problem 3: Sampling-based Tabular Q-Learning

```python
import random
random_num = random.random()
if random_num < eps:
    action = random.randint(0, len(q_vals[state])-1)
else:
    action = np.argmax(q_vals[state])
```

首先先sample action，有一定的機率會random選或是選argmax


```python
target = reward + gamma * np.max(q_vals[next_state])
q_vals[cur_state][action] = (1-alpha) * q_vals[cur_state][action] + alpha * target
```
接著implement Q learning update


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






