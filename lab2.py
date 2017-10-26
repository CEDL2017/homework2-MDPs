from misc import FrozenLakeEnv, make_grader
import pdb
env = FrozenLakeEnv()
print(env.__doc__)


# Some basic imports and setup
import numpy as np, numpy.random as nr, gym
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
np.set_printoptions(precision=3)

# Seed RNGs so you get the same printouts as me
env.seed(0); from gym.spaces import prng; prng.seed(10)
# Generate the episode
env.reset()
for t in range(100):
    env.render()
    a = env.action_space.sample()
    ob, rew, done, _ = env.step(a)
    if done:
        break
assert done
env.render();


class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means (used for plotting)
mdp = MDP( {s : {a : [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, env.nS, env.nA, env.desc)


print("mdp.P is a two-level dict where the first key is the state and the second key is the action.")
print("The 2D grid cells are associated with indices [0, 1, 2, ..., 15] from left to right and top to down, as in")
print(np.arange(16).reshape(4,4))
print("Action indices [0, 1, 2, 3] correspond to West, South, East and North.")
print("mdp.P[state][action] is a list of tuples (probability, nextstate, reward).\n")
print("For example, state 0 is the initial state, and the transition information for s=0, a=0 is \nP[0][0] =", mdp.P[0][0], "\n")
print("As another example, state 5 corresponds to a hole in the ice, in which all actions lead to the same state with probability 1 and reward 0.")
for i in range(4):
    print("P[5][%i] =" % i, mdp.P[5][i])


#Problem 1: implement value iteration
def value_iteration(mdp, gamma, nIt, grade_print=print):
    """
    Inputs:
        mdp: MDP
        gamma: discount factor
        nIt: number of iterations, corresponding to n above
    Outputs:
        (value_functions, policies)
        
    len(value_functions) == nIt+1 and len(policies) == nIt
    """
    grade_print("Iteration | max|V-Vprev| | # chg actions | V[0]")
    grade_print("----------+--------------+---------------+---------")
    Vs = [np.zeros(mdp.nS)] # list of value functions contains the initial value function V^{(0)}, which is zero
    pis = []
    for it in range(nIt):
        oldpi = pis[-1] if len(pis) > 0 else None # \pi^{(it)} = Greedy[V^{(it-1)}]. Just used for printout
        Vprev = Vs[-1] # V^{(it)}
        
        # Your code should fill in meaningful values for the following two variables
        # pi: greedy policy for Vprev (not V), 
        #     corresponding to the math above: \pi^{(it)} = Greedy[V^{(it)}]
        #     ** it needs to be numpy array of ints **
        # V: bellman backup on Vprev
        #     corresponding to the math above: V^{(it+1)} = T[V^{(it)}]
        #     ** numpy array of floats **
        
        # >>>>> Your code
        # hint: using 2 or 3 loop might be easy
        V = np.zeros(mdp.nS) # REPLACE THIS LINE WITH YOUR CODE
        pi = np.zeros(mdp.nS) # REPLACE THIS LINE WITH YOUR CODE 
        for s in range(mdp.nS):
            v_as = np.zeros(mdp.nA)
            for a in range(mdp.nA):
                for p in mdp.P[s][a]:
                    v_as[a] +=  p[0] * (p[2]+gamma*Vprev[p[1]]) #mdp.P: [probability, nextstate, reward]
            a_max = np.argmax(v_as)
            V[s] = v_as[a_max]
            pi[s] = a_max
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        
        max_diff = np.abs(V - Vprev).max()
        nChgActions="N/A" if oldpi is None else (pi != oldpi).sum()
        grade_print("%4i      | %6.5f      | %4s          | %5.3f"%(it, max_diff, nChgActions, V[0]))
        Vs.append(V)
        pis.append(pi)
    return Vs, pis

GAMMA = 0.95 # we'll be using this same value in subsequent problems


# The following is the output of a correct implementation; when
#   this code block is run, your implementation's print output will be
#   compared with expected output.
#   (incorrect line in red background with correct line printed side by side to help you debug)
expected_output = """Iteration | max|V-Vprev| | # chg actions | V[0]
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
  19      | 0.00003      |    0          | 0.531"""
Vs_VI, pis_VI = value_iteration(mdp, gamma=GAMMA, nIt=20, grade_print=make_grader(expected_output))

# Visualize the estimated values and corresponding policy

for (V, pi) in zip(Vs_VI[:10], pis_VI[:10]):
    plt.figure(figsize=(3,3))
    plt.imshow(V.reshape(4,4), cmap='gray', interpolation='none', clim=(0,1))
    ax = plt.gca()
    ax.set_xticks(np.arange(4)-.5)
    ax.set_yticks(np.arange(4)-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:4, 0:4]
    a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(-1, 0)}
    Pi = pi.reshape(4,4)
    for y in range(4):
        for x in range(4):
            a = Pi[y, x]
            u, v = a2uv[a]
            plt.arrow(x, y,u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1) 
            plt.text(x, y, str(env.desc[y,x].item().decode()),
                     color='g', size=12,  verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
    plt.grid(color='b', lw=2, ls='-')

plt.figure()
plt.plot(Vs_VI)
plt.xlabel("iterations")
plt.ylabel("Values")
plt.title("Values of different states");

# Problem 2: Policy Iteration

# Problem 2a: state value function
# solve the exact values with `np.linalg.solve`
def compute_vpi(pi, mdp, gamma):
    # use pi[state] to access the action that's prescribed by this policy
    # V[s] = P*(R + \gamma*V[s'])
    # => (I-\gamma*P)*V = P*R
    # solve linear matrix
    # (I-\gamma*P): (nS, nS) => a
    # P*R => b
    
    # >>>>> Your code
    a = np.zeros((mdp.nS, mdp.nS)) 
    b = np.zeros(mdp.nS) 
    for s in range(mdp.nS):
        a[s][s]=1 #idenity
        for p in mdp.P[s][pi[s]]:
            a[s][p[1]] -= gamma * p[0]
            b[s] += p[0] * p[2]
    V = np.linalg.solve(a, b)
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

# Problem 2b: state-action value function
def compute_qpi(vpi, mdp, gamma):
    # >>>>> Your code
    Qpi = np.zeros([mdp.nS, mdp.nA]) 
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for p in mdp.P[s][a]:
                Qpi[s][a] += p[0]*(p[2]+gamma*vpi[p[1]])
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

def policy_iteration(mdp, gamma, nIt, grade_print=print):
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.nS,dtype='int')
    pis.append(pi_prev)
    grade_print("Iteration | # chg actions | V[0]")
    grade_print("----------+---------------+---------")
    for it in range(nIt):
        # you need to compute qpi which is the state-action values for current pi
        #               and compute the greedily policy, pi, from qpi
        # >>>>> Your code (sample code are 3 lines)
        vpi = compute_vpi(pi_prev, mdp, gamma)
        qpi = compute_qpi(vpi, mdp, gamma)
        pi = np.argmax(qpi, axis=1)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        grade_print("%4i      | %6i        | %6.5f"%(it, (pi != pi_prev).sum(), vpi[0]))
        Vs.append(vpi)
        pis.append(pi)
        pi_prev = pi
    return Vs, pis

expected_output = """Iteration | # chg actions | V[0]
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
  19      |      0        | 0.53118"""

Vs_PI, pis_PI = policy_iteration(mdp, gamma=0.95, nIt=20, grade_print=make_grader(expected_output))

plt.plot(Vs_PI);
plt.xlabel("iterations")
plt.ylabel("Values")
plt.title("Values of different states");


# Problem 3: Sampling-based Tabular Q-Learning
from crawler_env import CrawlingRobotEnv

env = CrawlingRobotEnv()

print("We can inspect the observation space and action space of this Gym Environment")
print("-----------------------------------------------------------------------------")
print("Action space:", env.action_space)
print("It's a discrete space with %i actions to take" % env.action_space.n)
print("Each action corresponds to increasing/decreasing the angle of one of the joints")
print("We can also sample from this action space:", env.action_space.sample())
print("Another action sample:", env.action_space.sample())
print("Another action sample:", env.action_space.sample())
print("Observation space:", env.observation_space, ", which means it's a 9x13 grid.")
print("It's the discretized version of the robot's two joint angles")

env = CrawlingRobotEnv(
    render=True, # turn render mode on to visualize random motion
)

# standard procedure for interfacing with a Gym environment
cur_state = env.reset() # reset environment and get initial state
ret = 0.
done = False
i = 0
while not done:
    action = env.action_space.sample() # sample an action randomly
    next_state, reward, done, info = env.step(action)
    ret += reward
    cur_state = next_state
    i += 1
    #if not i%100:
    #    print(i) 
    if i == 1500:
        break # for the purpose of this visualization, let's only run for 1500 steps
        # also note the GUI won't close automatically

# you can close the visualization GUI with the following method 
env.close_gui()

from collections import defaultdict
import random

# dictionary that maps from state, s, to a numpy array of Q values [Q(s, a_1), Q(s, a_2) ... Q(s, a_n)]
#   and everything is initialized to 0.
q_vals = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))

print("Q-values for state (0, 0): %s" % q_vals[(0, 0)], "which is a list of Q values for each action")
print("As such, the Q value of taking action 3 in state (1,2), i.e. Q((1,2), 3), can be accessed by q_vals[(1,2)][3]:", q_vals[(1,2)][3])

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
    if random.random() < eps:
        action = random.randint(0, len(q_vals[state])-1)
    else:
        action = np.argmax(q_vals[state])
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    return action

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
##########################################################################
def q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward):
    """
    Inputs:
        gamma: discount factor
        alpha: learning rate
        q_vals: q value table
        cur_state: current state
        action: action taken in current state
        next_state: next state results from taking `action` in `cur_state`
        reward: reward received from this transition
    
    Performs in-place update of q_vals table to implement one step of Q-learning
    """
    # >>>>> Your code (sample code are 2 lines)
    target = reward + gamma *np.max(q_vals[next_state])
    q_vals[cur_state][action] = (1 -alpha) *q_vals[cur_state][action] + alpha *target
    # YOUR CODE HERE

# testing your q_learning_update implementation
dummy_q = q_vals.copy()
test_state = (0, 0)
test_next_state = (0, 1)
dummy_q[test_state][0] = 10.
dummy_q[test_next_state][1] = 10.
q_learning_update(0.9, 0.1, dummy_q, test_state, 0, test_next_state, 1.1)
tgt = 10.01
if np.isclose(dummy_q[test_state][0], tgt,):
    print("Test passed")
else:
    print("Q(test_state, 0) is expected to be %.2f but got %.2f" % (tgt, dummy_q[test_state][0]))
########################################################################
# now with the main components tested, we can put everything together to create a complete q learning agent

env = CrawlingRobotEnv() 
q_vals = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
gamma = 0.9
alpha = 0.1
eps = 0.5
cur_state = env.reset()

def greedy_eval():
    """evaluate greedy policy w.r.t current q_vals"""
    test_env = CrawlingRobotEnv(horizon=np.inf)
    prev_state = test_env.reset()
    ret = 0.
    done = False
    H = 100
    for i in range(H):
        action = np.argmax(q_vals[prev_state])
        state, reward, done, info = test_env.step(action)
        ret += reward
        prev_state = state
    return ret / H

for itr in range(300000):
    # YOUR CODE HERE
    # Hint: use eps_greedy & q_learning_update
    # >>>>> Your code (sample code are 4 lines)
    action = eps_greedy(q_vals, eps, cur_state)
    next_state, reward, done, info = env.step(action)
    q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward)
    cur_state = next_state
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    if itr % 50000 == 0: # evaluation
        print("Itr %i # Average speed: %.2f" % (itr, greedy_eval()))

# at the end of learning your crawler should reach a speed of >= 3

env = CrawlingRobotEnv(render=True, horizon=500)
prev_state = env.reset()
ret = 0.
done = False
while not done:
    action = np.argmax(q_vals[prev_state])
    state, reward, done, info = env.step(action)
    ret += reward
    prev_state = state
# you can close the visualization GUI with the following method 
env.close_gui()
