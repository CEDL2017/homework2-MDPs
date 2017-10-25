# Homework2 report

### 105061528 陳玉亭


### Overview:
>This homework is about value iteration, policy iteration and Q-learning.

### Implementation:
>I provide my solution in 'lab2.py' since it lags if I open jupyter remotely. However I have problem to setup the environment on my notebook. So, I give up to use ipython for above reasons. I also save the printed output in 'test.log' 

Value iteraion use Bellman's equation to update the value of states iteratively. And take greedy action to estimate largest value. </br>
	
	
	for s in range(mdp.nS): #all states
        v_as = np.zeros(mdp.nA) 
        for a in range(mdp.nA): # all acions 
            for p in mdp.P[s][a]: # pdf of (state,action)
                v_as[a] +=  p[0] * (p[2]+gamma*Vprev[p[1]]) #mdp.P: [prob, n_state, reward]
        V[s] = v_as[np.argmax(v_as)] #greedy action
        

	

Policy iteration estimate the value of policy by 2 steps: (1) update state value (V) (2) update state-action value (Q) by V, and also take greedy action to estimate largest Q. </br>
	
	#compute V :
	#(solve linear function V[s] = P*(R + \gamma*V[s']) by linalg.solve)		
	#compute Q :
	Qpi[s][a] += prob*(reward + gamma*vpi[nextState]) #psuedo code

Sampling-based Q-Learning
Use epsilon greedy sampling method to sample an action from value distribution. Save {S,a,r,S'} in the queue and provide Q-learning update.     

	# Q-learning updae 
	target = reward + gamma *np.max(q_vals[next_state])
    q_vals[cur_state][action] = (1 -alpha) *q_vals[cur_state][action] + alpha *target
