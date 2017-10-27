# Homework1 report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.
**************
Problem1: implement value iteration
In this problem, we are going to imnplement the value iteration.
Following instructions below, we will come out three loops with state, action and two-level dict in each.
We will update V and pi through the for loop and get the max and argmax which result in the expected output Vs and pis.

Example Code:
        V = np.array(Vprev) # REPLACE THIS LINE WITH YOUR CODE
        pi = np.zeros(mdp.nS) # REPLACE THIS LINE WITH YOUR CODE
        
        states = mdp.nS
        actions = mdp.nA
        
        for i in range(states):
            value = np.zeros(actions)
            for j in range(actions):
                for step in mdp.P[i][j]:
                    prob, nxt, rwd = step
                    value[j] = value[j] + prob*(rwd + GAMMA*Vprev[nxt])
                V[i] = np.max(value)
                
        for i in range(states):
            py = np.zeros(actions)
            for j in range(actions):
                for step in mdp.P[i][j]:
                    prob,nxt,rwd = step
                    py[j] = py[j] + prob*(rwd + GAMMA*Vprev[nxt])
                        
            pi[i] = np.argmax(py)
            
Iteration | max|V-Vprev| | # chg actions | V[0]
----------+--------------+---------------+---------
   0      | 0.80000      |  N/A          | 0.000
   1      | 0.60800      |    2          | 0.000
   2      | 0.51984      |    2          | 0.000
   3      | 0.39508      |    2          | 0.000
   4      | 0.30026      |    2          | 0.000
   5      | 0.25355      |    1          | 0.254
   6      | 0.10478      |    0          | 0.345
   7      | 0.09657      |    0          | 0.442
   8      | 0.03656      |    0          | 0.478
   9      | 0.02772      |    0          | 0.506
  10      | 0.01111      |    0          | 0.517
  11      | 0.00735      |    0          | 0.524
  12      | 0.00310      |    0          | 0.527
  13      | 0.00190      |    0          | 0.529
  14      | 0.00083      |    0          | 0.530
  15      | 0.00049      |    0          | 0.531
  16      | 0.00022      |    0          | 0.531
  17      | 0.00013      |    0          | 0.531
  18      | 0.00006      |    0          | 0.531
  19      | 0.00003      |    0          | 0.531
Test succeeded

PS：using [] in value iteration instead of () will derive a bad solution.
[] is used in describing the elemnt inside a list. 

*********************
In problem2, we are going to solve the exact value function with an arbitrary policy pi.
Following the instructions below, we know that b = P*R and a = a-gamma*P and by using the  np.linalg.solve to derive the exact values.
In 2b, we are going to compute qpi. Just simply using 3 for loops and feed the definition
Qπ(s,a)=∑s′P(s,a,s′)[R(s,a,s′)+γVπ(s′)]

     
    # >>>>> Your code
    a = np.zeros((mdp.nS, mdp.nS)) 
    b = np.zeros(mdp.nS)
    states = mdp.nS
   
    for i in range(states) :
        a[i][i] = 1
        for x in mdp.P[i][pi[i]]:
            prob, nxt, rwd = x
            b[i] = b[i] + prob*rwd
            a[i][nxt] = a[i][nxt] - gamma*prob
            
    V = np.linalg.solve(a,b)
    
    #V = np.zeros(mdp.nS) # REPLACE THIS LINE WITH YOUR CODE
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    return V
    
    expected_val = np.load('compute_vpi_result.npy')
    
policy = np.array([1, 0, 3, 3, 1, 3, 3, 1, 2, 2, 1, 1, 0, 3, 3, 3])
actual_val = compute_vpi(policy, mdp, gamma=GAMMA)
if np.all(np.isclose(actual_val, expected_val, atol=1e-4)):
    print("Test passed")
else:
    print("Expected: ", expected_val)
    print("Actual: ", actual_val)
   
    Test passed
    
    def compute_qpi(vpi, mdp, gamma):
    # >>>>> Your code
    Qpi = np.zeros([mdp.nS, mdp.nA]) # REPLACE THIS LINE WITH YOUR CODE
    for i in range(mdp.nS):
        for j in range(mdp.nA):
            for k in mdp.P[i][j]:
                prob, nxt, rwd = k
                Qpi[i][j] = Qpi[i][j] + prob*(rwd + gamma*vpi[nxt])
    
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    return Qpi

expected_Qpi = np.array([[  0.38 ,   3.135,   1.14 ,   0.095],
       [  0.57 ,   3.99 ,   2.09 ,   0.95 ],
       [  1.52 ,   4.94 ,   3.04 ,   1.9  ],
       [  2.47 ,   5.795,   3.23 ,   2.755],
       [  3.8  ,   6.935,   4.56 ,   0.855],
       [  4.75 ,   4.75 ,   4.75 ,   4.75 ],
       [  4.94 ,   8.74 ,   6.46 ,   2.66 ],
       [  6.65 ,   6.65 ,   6.65 ,   6.65 ],
       [  7.6  ,  10.735,   8.36 ,   4.655],
       [  7.79 ,  11.59 ,   9.31 ,   5.51 ],
       [  8.74 ,  12.54 ,  10.26 ,   6.46 ],
       [ 10.45 ,  10.45 ,  10.45 ,  10.45 ],
       [ 11.4  ,  11.4  ,  11.4  ,  11.4  ],
       [ 11.21 ,  12.35 ,  12.73 ,   9.31 ],
       [ 12.16 ,  13.4  ,  14.48 ,  10.36 ],
       [ 14.25 ,  14.25 ,  14.25 ,  14.25 ]])

Qpi = compute_qpi(np.arange(mdp.nS), mdp, gamma=GAMMA)
if np.all(np.isclose(expected_Qpi, Qpi, atol=1e-4)):
    print("Test passed")
else:
    print("Expected: ", expected_Qpi)
    print("Actual: ", Qpi)
    
    Test passed
    Iteration | # chg actions | V[0]
----------+---------------+---------
   0      |      1        | -0.00000
   1      |      9        | 0.00000
   2      |      2        | 0.39785
   3      |      1        | 0.45546
   4      |      0        | 0.53118
   5      |      0        | 0.53118
   6      |      0        | 0.53118
   7      |      0        | 0.53118
   8      |      0        | 0.53118
   9      |      0        | 0.53118
  10      |      0        | 0.53118
  11      |      0        | 0.53118
  12      |      0        | 0.53118
  13      |      0        | 0.53118
  14      |      0        | 0.53118
  15      |      0        | 0.53118
  16      |      0        | 0.53118
  17      |      0        | 0.53118
  18      |      0        | 0.53118
  19      |      0        | 0.53118
Test succeeded
***************

 In problem3, we are going to do the Taabular Q-Learing
 We have to define the eps_greedy first.
 We will have q table, epsilon and state as  input and following the instructions we know that we have to create a rand by    random.random(), however i tried random.uniform in between 0.0,1.0, and it works as well.
 Next, we give it a decision rule, if random is smaller than the designated epsilon, it will start to explore ramdomly.
Else, it will find the argmax in q table.



def eps_greedy(q_vals, eps, state):
    """
    Inputs:
        q_vals: q value tables
        eps: epsilon
        state: current state
    Outputs:
        random action with probability of eps; argmax Q(s, .) with probability of (1-eps)
    """
    # you might want to use random.random() to implement random exploration
    #   number of actions can be read off from len(q_vals[state])
    import random
    # >>>>> Your code
    #rand = random.random()
    rand = random.uniform(0.0,1.0)
    
    if rand < eps:
        action = random.randint(0, len(q_vals[state])-1)
    else:
        action = np.argmax(q_vals[state])
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    return int(action)

# test case 1
dummy_q = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
test_state = (0, 0)
dummy_q[test_state][0] = 10.
trials = 100000
sampled_actions = [
    int(eps_greedy(dummy_q, 0.3, test_state))
    for _ in range(trials)
]
freq = np.sum(np.array(sampled_actions) == 0) / trials
tgt_freq = 0.3 / env.action_space.n + 0.7
if np.isclose(freq, tgt_freq, atol=1e-2):
    print("Test1 passed")
else:
    print("Test1: Expected to select 0 with frequency %.2f but got %.2f" % (tgt_freq, freq))
    
# test case 2
dummy_q = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
test_state = (0, 0)
dummy_q[test_state][2] = 10.
trials = 100000
sampled_actions = [
    int(eps_greedy(dummy_q, 0.5, test_state))
    for _ in range(trials)
]
freq = np.sum(np.array(sampled_actions) == 2) / trials
tgt_freq = 0.5 / env.action_space.n + 0.5
if np.isclose(freq, tgt_freq, atol=1e-2):
    print("Test2 passed")
else:
    print("Test2: Expected to select 2 with frequency %.2f but got %.2f" % (tgt_freq, freq))

 Test1 passed
 Test2 passed
 
 ***
 
 Next, we will update the Q learning procedure. With the definition provided below
target(s′)=R(s,a,s′)+γmaxa′Qθk(s′,a′)
Qk+1(s,a)←(1−α)Qk(s,a)+α[target(s′)]

Simply using the sample code, and it works.
q_vals[cur_state][action] = (1-alpha)*q_vals[cur_state][action]+alpha*(reward+gamma*np.max(q_vals[next_state])) 
cur_state = next_state

Finally we are going to test the Robot

1.First call the greedy function

action = eps_greedy(q_vals, eps, cur_state)

2.Next, we will use the q_learning update

next_state, reward, _, _ = env.step(action)
q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward)

3.Finally, update the cur_state

cur_state = next_state


Itr 0 # Average speed: 0.05
Itr 50000 # Average speed: 2.03
Itr 100000 # Average speed: 3.37
Itr 150000 # Average speed: 3.37
Itr 200000 # Average speed: 3.37
Itr 250000 # Average speed: 3.37








 
 
 


