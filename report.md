# Homework1 report

謝廷翊 106062612

this assigiment was very interesting , we can realize how to produce Markov Model .
i think this policy decision can apply in many different research.
  
code 

Value Iteration
  
        pi = np.zeros(mdp.nS) # pi: one action per state (length 16)
        V = np.zeros(mdp.nS) # V: one value per state (length 16)
        
        for state in range(mdp.nS):
            tmp = np.zeros(4)
            for action in range(mdp.nA):
                value = 0
                for action_choose in mdp.P[state][action]:
                    value += action_choose[0]*(action_choose[2]+gamma*Vprev[action_choose[1]])
                tmp[action]=value
            pi[state] = np.argmax(tmp)
            V[state] = np.max(tmp)
        
        
Policy Iteration
 1. compute_vpi that computes the state-value function for an arbitrary policy
                 
    a = np.identity(mdp.nS)
    b = np.zeros(mdp.nS) 
    V = np.zeros(mdp.nS) # REPLACE THIS LINE WITH YOUR CODE
    
    for state in range(len(pi)):
        tmp=mdp.P[state][pi[state]]
        for option in range(len(tmp)):
            a[state][tmp[option][1]] -= gamma*tmp[option][0]
            b[state] += tmp[option][0]*tmp[option][2]
    V = np.linalg.solve(a,b)
    
        
 2 . compute_qpi that compute the state-action value function
  
    
    for state in range(mdp.nS):
        for action in range(mdp.nA):
            for action_chose in mdp.P[state][action]:
                Qpi[state][action] += action_chose[0]*(action_chose[2]+gamma*vpi[action_chose[1]])
                
                
  3.run the poolicy iteration that compute qpi which is the state-action values for current pi and compute the greedily policy, pi, from     qpi
    
          vpi = compute_vpi(pi_prev, mdp, gamma)
          Qpi = compute_qpi(vpi, mdp, gamma)
          pi = np.argmax(Qpi, axis=1)
    
Tabular Q-Learning
  
  1.eps_greedy that get random action with eps
  
    random_action=random.randrange(len(q_vals[state]))
    action = np.argmax(q_vals[state])
    if random.random() < eps:
         action = random_action
  
  2.q_learning_update that update the q_vals table to implement one step of Q-learning
  
    target = reward + gamma * np.max(q_vals[next_state])
    q_vals[cur_state][action] = ((1 - alpha) * q_vals[cur_state][action]) + (alpha * target)
    
  3.put everything together to create a complete q learning agent
  
    action = eps_greedy(q_vals, eps, cur_state)
    next_state, reward, done, info = env.step(action)
    q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward)
    cur_state = next_state
