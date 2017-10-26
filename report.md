# Homework1 report

## Value Iteration:
Implement the value iteration formula:
![](https://github.com/hellochick/homework2-MDPs/blob/master/imgs/value_iteration.png)

We go through every state, and then we first get every possible next state. We sum over every possible next state, and figure out the max value of actions to update value in this state.
```python
for s in range(mdp.nS):
    action_v = np.zeros(mdp.nA)
    for a in range(mdp.nA):
      value = 0
      for prob, next_state, reward in mdp.P[s][a]:
          value += prob * (reward + gamma * Vprev[next_state])
                
      action_v[a] = value
                
    V[s] = np.max(action_v)
```
## Policy iteration:

## 
