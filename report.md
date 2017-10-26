# Homework2 report
## 103062108 鄭安傑

## Value Iteration
Implement a greedy value interation update based on formula.
```python
V = np.zeros(mdp.nS)
pi = np.zeros((mdp.nS), dtype=np.int)
        
for state in range(mdp.nS):
    V_list = []
    for action in range(mdp.nA):
        candidates = mdp.P[state][action]
        v = 0
        for probability, nextstate, reward in candidates:
            v += probability * (reward + gamma * Vprev[nextstate])
        V_list.append(v)
    V[state] = np.max(V_list)
    pi[state] = np.argmax(V_list)
```
## Policy Iteration
Solve the exact values with `np.linalg.solve`
```python
a = np.zeros((mdp.nS, mdp.nS)) 
b = np.zeros(mdp.nS) 
for state in range(mdp.nS):
for probability, nextstate, reward in mdp.P[state][pi[state]]:
    a[state][nextstate] += probability
    b[state] += probability * reward
V = np.linalg.solve(np.eye(len(a)) - gamma*a, b)
```
Basically based on the given pseudo code. For some unknown reason, if I multiply `probability` and `reward` outside the for loop, the value would be wrong. So I multiply inside the for loop.
Compute the Q-function. 
```python
Qpi = np.zeros([mdp.nS, mdp.nA])
for state in range(mdp.nS):
    for action in range(mdp.nA):
        for probability, nextstate, reward in mdp.P[state][action]:
            Qpi[state][action] += probability * (reward + gamma * vpi[nextstate])
```
Below is the main function of policy iteration, which updates the new policy through Q-function.
```python
vpi = compute_vpi(pi_prev, mdp, gamma)
qpi = compute_qpi(vpi, mdp, gamma)
pi = np.array([np.argmax(q[:,]) for q in qpi])
```

## Sampling-based Tabular Q-Learning
Randomly sample an action.
```python
if np.random.random() < eps:
    action = np.random.randint(0, len(q_vals[state]))
else:
    action = np.argmax(q_vals[state])
```
Update Q-function
```python
target = reward + gamma * np.max(q_vals[next_state])
q_vals[cur_state][action] = (1-alpha) * q_vals[cur_state][action] + alpha * target
```
The main part of updating iteration
```python
for itr in range(300000):
    action = eps_greedy(q_vals, eps, cur_state)
    next_state, reward, done, info = env.step(action)
    q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward)
    cur_state = next_state
```
