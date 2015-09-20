# Homework1 report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.
## Value iteration
#### Pseudo
Initialize $V^{(0)}(s)=0$, for all $s$

For $i=0, 1, 2, \dots$
- $V^{(i+1)}(s) = \max_a \sum_{s'} P(s,a,s') [ R(s,a,s') + \gamma V^{(i)}(s')]$, for all $s$

-  $\pi^{(0)}, \pi^{(1)}, \dots, \pi^{(n-1)}$, where
       $$\pi^{(i)}(s) = \arg \max_a \sum_{s'} P(s,a,s') [ R(s,a,s') + \gamma V^{(i)}(s')]$$

 return two lists: $[V^{(0)}, V^{(1)}, \dots, V^{(n)}]$ and $[\pi^{(0)}, \pi^{(1)}, \dots, \pi^{(n-1)}]$
 
#### Implementation
`V = np.array(Vprev)
        pi = np.zeros(mdp.nS)
        for i in range(mdp.nS):
            val_tmp = np.zeros(mdp.nA)
            policy_tmp = np.zeros(mdp.nA)
            for j in range(mdp.nA):
                for _ in mdp.P[i][j]:
                    prob, next_state, reward = _
                    val_tmp[j] += prob * (reward + GAMMA * Vprev[next_state])
                    policy_tmp[j] += prob * (reward + GAMMA * Vprev[next_state])
                V[i] = np.max(val_tmp)
                pi[i] = np.argmax(policy_tmp)`

## Policy iteration
### State value function
#### Pseudo

- $$V^{\pi}(s) = \sum_{s'} P(s,\pi(s),s')[ R(s,\pi(s),s') + \gamma V^{\pi}(s')]$$
- a = (I-\gamma*P): (nS, nS)
- b = P*R

#### Implementation

`for i in range(mdp.nS):
        a[i][i] = 1
        for _ in mdp.P[i][pi[i]]:
            prob, next_state, reward= _
            b[i] = b[i] + prob * reward
            a[i][next_state] = a[i][next_state] - gamma * prob
    V = np.linalg.solve(a, b)`
    
### State-action value function
#### Pseudo
- $$Q^{\pi}(s, a) = \sum_{s'} P(s,a,s')[ R(s,a,s') + \gamma V^{\pi}(s')]$$

#### Implementation
`    
    for i in range(mdp.nS):
        for j in range(mdp.nA):
            for _ in mdp.P[i][j]:
                prob, next_state, reward = _
                Qpi[i][j] = Qpi[i][j] + prob * (reward+gamma*vpi[next_state])`
                
### Iteration
`vpi=compute_vpi(pi_prev, mdp, gamma=gamma)
qpi=compute_qpi(vpi, mdp, gamma)
pi = qpi.argmax(axis = 1)`
                
                
## Tabular Q-learning
### Eps-greedy
#### pseudo
- random exploration (prob: eps)

#### Implementation
`action = 0
if random.random() < eps: # random exploration
    action = random.randint(0,len(q_vals[state])-1)
else:
    action = np.argmax(q_vals[state])`

### Update Q learning
#### Pseudo
- $$\textrm{target}(s') = R(s,a,s') + \gamma \max_{a'} Q_{\theta_k}(s',a')$$
- $$Q_{k+1}(s,a) \leftarrow (1-\alpha) Q_k(s,a) + \alpha \left[ \textrm{target}(s') \right]$$

#### Implementation
`target = reward + gamma * q_vals[next_state][np.argmax(q_vals[next_state])]
q_vals[cur_state][action] = (1-alpha) * q_vals[cur_state][action] + alpha * target`

### Iteration
`action = eps_greedy(q_vals, eps, cur_state)
next_state, reward = env.step(action)[0:2]
q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward)
cur_state = next_state`

#### Others
CrawlingRobotEnv() always occurs error, haven't found out solution.