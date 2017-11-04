# Homework2-MDPs report

## 103061148 鄭欽安

## Value iteration
With known MDPs, implement Value iteration.   
* Use immediate reward and one step look ahead, previous Value function to update the current Value function ,then return updated Value function and greedy policy accroding to Value function until converage.  
```python
V = np.zeros(mdp.nS)
pi = np.zeros(mdp.nS)
  for state in range(mdp.nS):
    max_val = -1
    argmax_action = -1
    for action in range(mdp.nA):
      val = 0
      for prob, next_state, reward in mdp.P[state][action]:
        val += prob*(reward + gamma*Vprev[next_state])
        if val > max_val:
          max_val = val
          argmax_action = action
          V[state] = max_val
          pi[state] = argmax_action
```

## Policy iteration
With known MDPs, implement policy iteration.
* Compute current Value function through current policy function. By soving linear equation, we will get Value function.    
```python
a = np.identity(mdp.nS)
b = np.zeros(mdp.nS) 
for state in range(mdp.nS):        
  for prob, next_state, reward in mdp.P[state][pi[state]]:
    b[state] += prob * reward
    a[state][next_state] = a[state][next_state] - gamma * prob        
V = np.linalg.solve(a, b)
``` 
* Compute Q-function through current Value function, Q-function is state-action values and Value function is state values.
```python
Qpi = np.zeros([mdp.nS, mdp.nA])
for state in range(mdp.nS):
  for action in range(mdp.nA):
    for prob, next_state, reward in mdp.P[state][action]:
      Qpi[state][action] += prob*(reward + gamma*vpi[next_state])
 ```  
* Update new policy through greedy Q-function. Use greedy algoritm on previous calculated Q-function to update policy function.
```ruby
pi = np.argmax(qpi, axis=1)
```
  
## Tabular Q-Learning
Without whole transition model, implement sample-based Q-learning.
* eps greedy, use it to sample action with current state and current Q-function  
```python
random_action = math.floor(random.random() * len(q_vals[state]))
max_action = np.argmax(q_vals[state])
if random.random() >= eps:
  action = max_action
else: 
  action = random_action
```
* Q-learning update, use it to update Q-function   
```python
target = reward + gamma*max(q_vals[next_state])
q_vals[cur_state][action] = (1-alpha)*q_vals[cur_state][action] + alpha*target
```
* Q-learing update iteration  
```python
for itr in range(300000):
  action = eps_greedy(q_vals, eps, cur_state)
  next_state, reward, done, info = env.step(action)
  q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward)
  cur_state = next_state
```

