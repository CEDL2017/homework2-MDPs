# 何品萱 (106062553)
## Homework2 report


#### Algorithms and code details
Problem 1: implement value iteration

<p align="center"><img src="imgs/value iteration1.jpg" width=50% /></p>
<p align="center"><img src="imgs/value iteration2.jpg" width=60%/></p>

```
  V = np.zeros(mdp.nS)
  pi = np.zeros(mdp.nS)
        
  for state in range(mdp.nS):          
    V_a = np.zeros(mdp.nA)   
    for direct in range(mdp.nA):
      for P, next_s, r in mdp.P[state][direct]:   #mdp.P[state][action] (probability, nextstate, reward)
        V_a[direct] += P * (r + gamma * Vprev[next_s])
    V[state]=np.amax(V_a)
    pi[state]=np.argmax(V_a)
```

Problem 2: Policy Iteration

state value function
<p align="center"><img src="imgs/Policy Iteration1.jpg" width=40%/></p>

`numpy.linalg.solve`   solve a linear matrix equation

```
    for state in range(mdp.nS):
        a[state][state] = 1 #identity matrix
        for P, next_s, r in mdp.P[state][pi[state]]:
            a[state][next_s] = a[state][next_s] -gamma * P
            b[state] = b[state] + P * r
    V = np.linalg.solve(a, b)   # aV = b
```

state-action value function
<p align="center"><img src="imgs/Policy Iteration2.jpg" width=40%/></p>

```
    for s in range(mdp.nS):
        for a in range(mdp.nA):   #West, South, East and North
            for P, next_s, r in mdp.P[s][a]: # (probability, nextstate, reward)
                Qpi[s][a] = Qpi[s][a] + P * (r + gamma * vpi[next_s])
```

Problem 3: Sampling-based Tabular Q-Learning
<p align="center"><img src="imgs/Q learning update.jpg" width=40%/></p>

eps_greedy

```
action = random.randrange(len(q_vals[state])) if random.random() < eps else np.argmax(q_vals[state])
```

implement Q learning update

```
    target = reward + gamma * np.max(q_vals[next_state])
    q_vals[cur_state][action] = (1-alpha) * q_vals[cur_state][action] + alpha * target
```

put everything together to create a complete q learning agent

```
    action = int(eps_greedy(q_vals, eps, cur_state))
    next_s, reward, done, info = env.step(action)
    q_learning_update(gamma, alpha, q_vals, cur_state, action, next_s, reward)
    cur_state = next_s
```

#### Results
<p><img src="imgs/Average speed.jpg" width=40%/></p>
