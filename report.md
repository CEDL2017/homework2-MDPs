# Homework1 report
### Problem 1
```python
V = np.zeros(mdp.nS)
        pi = np.zeros(mdp.nS)
        for s in range(mdp.nS):
            v_action = np.zeros(mdp.nA)
            for a in range (mdp.nA):
                for p in mdp.P[s][a]:
                    v_action[a] += p[0] * (Vprev[p[1]] * gamma + p[2])
            argmax_a = np.argmax(v_action)
            V[s] = v_action[argmax_a]
            pi[s] = argmax_a
```
根據value iteration的公式，先初始化所有的State跟pi(action)
然後對每個state都用Greedy algorithm找出最好的action跟reward，
並更新V跟pi。

### Problem 2a
填空compute_vpi function
```python
    a = np.zeros((mdp.nS, mdp.nS)) 
    b = np.zeros(mdp.nS) 
    V = np.zeros(mdp.nS)
    for s in range(mdp.nS):
        a[s][s] = 1;
        for p in mdp.P[s][pi[s]]:
            b[s] += p[0] * p[2] 
            a[s][p[1]] -= gamma * p[0]
    V = np.linalg.solve(a, b)
```
這題主要就是把原本的式子寫出來然後直接使用linalg.solve函數，
沒有特別困難的地方。

### Problem 2b
```python

def compute_qpi(vpi, mdp, gamma):
    # >>>>> Your code
    Qpi = np.zeros([mdp.nS, mdp.nA]) # REPLACE THIS LINE WITH YOUR CODE
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for p in mdp.P[s][a]:
                Qpi[s][a] += p[0] * (gamma * vpi[p[1]] + p[2])
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    return Qpi
```
先寫出compute qpi的function，
藉由助教給的公式，
可以直接法sigma寫成迴圈，
跟value iteration最大的差別就是沒有argmax。
```python
        vpi = compute_vpi(pi_prev, mdp, gamma)
        qpi = compute_qpi(vpi, mdp, gamma)
        pi = qpi.argmax(axis = 1)
```
然後直接使用前面的compute_vpi跟compute_qpi就可以得到結果。
### Problem 3
```python
def eps_greedy(q_vals, eps, state):
    """
    Inputs:
        q_vals: q value tables
        eps: epsilon
        state: current state
    Outputs:
        random action with probability of eps; argmax Q(s, .) with probability of (1-eps)
    """
    # you might want to use random.random() to implement random exploration
    #   number of actions can be read off from len(q_vals[state])
    import random
    
    # >>>>> Your code
    action = 0
    if random.random() < eps:
        action = int(random.random()*4)
    else:
        action = np.argmax(q_vals[state])
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    return action
```
完成eps_greedy，直接使用random來寫，找藥random的值超過eps就greedy，沒超過就random，
這邊寫```random.random()*4```的目的是action會從1~4，如果不乘4就會永遠是1。
```python
def q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward):
    """
    Inputs:
        gamma: discount factor
        alpha: learning rate
        q_vals: q value table
        cur_state: current state
        action: action taken in current state
        next_state: next state results from taking `action` in `cur_state`
        reward: reward received from this transition
    
    Performs in-place update of q_vals table to implement one step of Q-learning
    """
    # >>>>> Your code (sample code are 2 lines)
    target = reward + gamma * q_vals[next_state][np.argmax(q_vals[next_state])]
    q_vals[cur_state][action] = (1-alpha) * q_vals[cur_state][action] + alpha * target 
    # YOUR CODE HERE
```
完成q-learning基本上一樣只是簡單吧公式打成code而已沒有特別困難的地方。
```python
    action = eps_greedy(q_vals, eps, cur_state)
    next_state, reward = env.step(action)[0:2]
    q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward)
    cur_state = next_state
```
直接使用並把參數叫進去而已。
TA: try to elaborate the algorithms that you implemented and any details worth mentioned.
