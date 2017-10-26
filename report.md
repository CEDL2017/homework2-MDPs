# Homework1 report

這次作業基本上搞清楚MDP class裡面的elements後,照著公式實做即可

# Problem 1: implement value iteration


        V = np.zeros(mdp.nS)
        pi = np.zeros(mdp.nS)
        
        for s in range(mdp.nS):
            V_act = np.zeros(mdp.nA)
            for a in range(mdp.nA):
                for P, NS, R in mdp.P[s][a]: # prob.,next stage,reward
                    V_act[a] += P * ( R + gamma * Vprev[NS])
            V[s] = np.max(V_act)
            pi[s] = np.argmax(V_act)


# Problem 2: Policy Iteration

# a).state value function
    a = np.eye(mdp.nS) 
    b = np.zeros(mdp.nS) 
    V = np.zeros(mdp.nS) 
    for s in range(mdp.nS):
        for P, NS, R in mdp.P[s][pi[s]]: # prob.,next stage,reward
            a[s][NS] -= P*gamma
            b[s] += P * R
    V = np.linalg.solve(a, b)

# b).state-action value function
    Qpi = np.zeros([mdp.nS, mdp.nA]) 
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for P, NS, R in mdp.P[s][a]:
                Qpi[s][a] += P * (R + gamma * vpi[NS])

# Problem 3: Sampling-based Tabular Q-Learning

# greedy
    x = random.uniform(0, 1)
    if x < eps:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_vals[state])

# Q learning update
    target = reward + gamma * max(q_vals[next_state][0], q_vals[next_state][1], q_vals[next_state][2], q_vals[next_state][3])
    q_vals[cur_state][action] = (1-alpha) * q_vals[cur_state][action] + alpha * target

