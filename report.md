# Homework2 report: Markov Decision Processes (MDPs) 
## 靳文綺 <span style="color:red">(106062563)</span>

## Overview
In this homework, there are mainly three problems that we want to solve:
* <b>Problem 1: implement value iteration</b> 
* <b>Problem 2: Policy Iteration</b>
	* Problem 2a: state value function
	* Problem 2b: state-action value function
* <b>Problem 2: Policy Iteration</b>
* <b>Problem 3: Sampling-based Tabular Q-Learning</b>



## Installation
* Python 3.6.2
* numpy
* matplotlib
* ipython
* [OpenAI gym](https://github.com/openai/gym)
	```
	pip install gym
	```
	
	
## Detail
<b>1. Value Iteration</b>
We use `Bellman Equation` to update the value. We calculate all the expected values and each (s,a) will be many possible state.
Finally we can get the value which is the best in current state. And you can use this value to calculate the policy.

```
    for it in range(nIt):
        oldpi = pis[-1] if len(pis) > 0 else None 
        Vprev = Vs[-1] 
        
        nA = mdp.nA # the number of actions
        nS = mdp.nS # the number of states
        pi = np.zeros(nS) # pi: one action per state (length 16)
        V = np.zeros(nS) # V: one value per state (length 16)
        
        for state in range(nS):
            expected_values = np.zeros(nA) # Each (s,a) per action value
            
            # Calculate all expected values
            for action in range(nA):
                expected_value = 0
                # Each (s,a) will be many possible s'
                for s_next in mdp.P[state][action]:
                    expected_value += s_next[0]*(s_next[2] + gamma*Vprev[s_next[1]])
                expected_values[action] = expected_value
            
            # Choose the best action
            pi[state] = np.argmax(expected_values)
            V[state] = np.max(expected_values)
            
        max_diff = np.abs(V - Vprev).max()
        nChgActions="N/A" if oldpi is None else (pi != oldpi).sum()
        grade_print("%4i      | %6.5f      | %4s          | %5.3f"%(it, max_diff, nChgActions, V[0]))
        Vs.append(V)
        pis.append(pi)
    return Vs, pis
```

<b>2. Policy Iteration</b>
Using `Bellman Equation` to update the value, finally we can get the value of current policy. We compute the `state value function` and the `state-action value function`.
	* state value function
	```
	def compute_vpi(pi, mdp, gamma):
		nA = mdp.nA # The number of actions
		nS = mdp.nS # The number of states
		
		a = np.zeros((nS, nS)) # aV = b, a(nS,nS)
		b = np.zeros(nS) # b(nS,1)
		
		for state in range(nS):
			a[state][state] += 1 # Vpi(state)
			action = pi[state] # Get action from policy
			b_state = 0 # The entry in the b vector
			for s_next in mdp.P[state][action]:
				b_state += s_next[0]*s_next[2]
				a[state][s_next[1]] -= gamma * s_next[0] 
			b[state] = b_state    						
		V = np.linalg.solve(a, b)
		
		return V
	```

	* state-action value function
	```
	def compute_qpi(vpi, mdp, gamma):
		nA = mdp.nA # The number of actions
		nS = mdp.nS # The number of states
		Qpi = np.zeros((nS, nA))
		
		for state in range(nS):
			expected_values = np.zeros(nA)
			# Calculate all the expected values
			for action in range(nA):
				expected_value = 0
				# Each (s,a) will be many possible s'
				for s_next in mdp.P[state][action]:
					expected_value += s_next[0]*(s_next[2] + gamma*vpi[s_next[1]])
				Qpi[state][action] = expected_value
		  
		return Qpi
	```

<b>3. Sampling-based Tabular Q-Learning</b>
Q-Learning works by learning an action-value function that ultimately gives the expected utility of taking a given action in the state and following the optimal policy thereafter.
There will have the parameter called epsilon-greedy, which can control the probability that will choose the action of the best value.
There is the algorithm:
![Q-Learning algorithm](Q-Learning.png)

	* epsilon-greedy
	```
	import random

    x = random.uniform(0, 1)
    if x < eps:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_vals[state])
    
    return action
	```
	
	* Q-Learning update
	```
	q_target = reward + gamma * np.max(q_vals[next_state])
    q_vals[cur_state][action] = (1-alpha) * q_vals[cur_state][action] + alpha * q_target
	```