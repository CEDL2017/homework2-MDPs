# Homework2 Report: MDPs

### 106065507 徐慧文
## Overview
We solve MDPs with finte state and action space via value iteration, policy iteration, and tabular Q-learning.
## Environment
* Python 3.5.3
* OpenAI gym
* numpy
* matplotlib
* ipython
## Implementation
### Problem 1: value iteration
Value iteration is at each iteration of evaluation of all states of the environment, we increment the value of a state depending on values of neighbor states, doing this until all the environment is covered. Using Bellman optimality equation to update
equation for V.
```
Vnext = []#one dim, 16 index
PI_list=[]
for s_cn in range(0,len(mdp.P)): # state's counter
	value_array = []
    for a_cn in range(0,len(mdp.P[s_cn])):# action's counter
    	dir_value = 0
        for infor_cn in range(0,len(mdp.P[s_cn][a_cn])): #each action's information
        	P, Snext,R = mdp.P[s_cn][a_cn][infor_cn]
            dir_value += P*( R + GAMMA*Vprev[Snext] )

        value_array.append(dir_value)

    Vnext.append(max(value_array)) 
    PI_list.append(np.argmax(value_array))
```
### Problem 2: Policy Iteration

Policy Iteration includes "Policy Evaluation" and "Policy Improvement".
##### 2a- Policy Evaluation (state value function)
The actions taken at each state are fixed. We have to try all possible state and action pairs from our transition probability Ps,a,s' based on the predetermined policy "pi".
```
# a is a  2-D array with ones on the diagonal and zeros elsewhere
a = np.identity(mdp.nS) 
b = np.zeros(mdp.nS)
V = np.zeros(mdp.nS)
for s_cn in range(0,mdp.nS): # state's counter
	a_cn = pi[s_cn]
    for infor_cn in range(0,len(mdp.P[s_cn][a_cn])): 
    	P, Snext,R = mdp.P[s_cn][a_cn][infor_cn]
        a[s_cn][Snext] -=gamma*P
        b[s_cn] += P*R  
        V = np.linalg.solve(a, b)
```
##### 2b- Policy Improvement (action value function)
Update policy using one-step look-ahead with resulting converged utilities as future values. 

```
Qpi = np.zeros([mdp.nS, mdp.nA])
for s_cn in range(0,mdp.nS): # state's counter
	for a_cn in range(0, mdp.nA):
    tempValue = 0
    for infor_cn in range(len(mdp.P[s_cn][a_cn])):
    	P, Snext,R = mdp.P[s_cn][a_cn][infor_cn]
        tempValue += P*( R + gamma*vpi[Snext] )
    Qpi[s_cn][a_cn] = tempValue
```

##### 2c- Policy Iteration
It is guaranteed to converge and at convergence, the current policy 
and its value function  are the optimal policy and the optimal value function.
```
vpi = compute_vpi(pi_prev, mdp, gamma=GAMMA)
Qpi = compute_qpi(vpi, mdp, gamma=GAMMA)
pi = np.argmax(Qpi, axis=1)
```
### Problem 3: Sampling-based Tabular Q-Learning
The algorithm converges with probability 1 to a close approximation of the action-value function for an arbitrary target policy. Q-Learning learns the optimal policy even when actions are selected according to a more exploratory or even random policy. 
##### 3-1: eps_greedy
Choose an action for that state based on one of the action selection policies. If random action with probability of eps, the actions is determined by the lenth of Qvals[current state]; if random action with probability of (1-eps), We will choose the max action number of the state.
```
def eps_greedy(q_vals, eps, state):
    action_numbers = q_vals[state]
    if (random.random()< eps) :
        action = random.randrange(len(q_vals[state]))
    else:
        action = np.argmax(action_numbers)
    return action
```
##### 3-2: q-learning_update
Update the Q-value for the state using the observed reward and the maximum reward possible for the next state
```
target = reward+ gamma*max(q_vals[next_state])
q_vals[cur_state][action] = (1-alpha)*q_vals[cur_state][action]+ alpha*target
```
##### 3-3: completely q learning agent
Set the state to the new state, and repeat the process until a terminal state is reached.
```
action = eps_greedy(q_vals, eps, cur_state)
next_state, reward, done, info = env.step(action)
q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward)
cur_state = next_state
```

