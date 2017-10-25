# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:14:54 2017

@author: CHADSHEEP
"""
from misc import FrozenLakeEnv, make_grader
import numpy as np, numpy.random as nr, gym
import matplotlib.pyplot as plt

class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means (used for plotting)


#%% Policy Iteratioin I Comput Vpi
def compute_vpi(pi, mdp, gamma):
    # use pi[state] to access the action that's prescribed by this policy
    # V[s] = P*(R + \gamma*V[s'])
    # => (I-\gamma*P)*V = P*R
    # solve linear matrix
    # (I-\gamma*P): (nS, nS) => a
    # P*R => b
    
    tmp = np.zeros((mdp.nS, mdp.nS)) 
    b = np.zeros(mdp.nS) 
    for state, actions in mdp.P.items():
        ac = pi[state]
        for action in actions[ac]:
            prob = action[0]
            nextState = action[1]
            r = action[2]
            b[state] += prob*r
            tmp[state, nextState] += gamma*prob        
    a = np.eye(mdp.nS) - tmp
    V = np.linalg.solve(a,b)
    return V

# Create env
env = FrozenLakeEnv()
print(env.__doc__)

# test vpi
GAMMA = 0.95 # we'll be using this same value in subsequent 
mdp = MDP( {s : {a : [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, env.nS, env.nA, env.desc)
expected_val = np.load('compute_vpi_result.npy')
policy = np.array([1, 0, 3, 3, 1, 3, 3, 1, 2, 2, 1, 1, 0, 3, 3, 3])
actual_val = compute_vpi(policy, mdp, gamma=GAMMA)
if np.all(np.isclose(actual_val, expected_val, atol=1e-4)):
    print("Test passed")
else:
    print("Expected: ", expected_val)
    print("Actual: ", actual_val)


#%% Policy Iteratioin II Comput Qpi
def compute_qpi(vpi, mdp, gamma):
    Qpi = np.zeros([mdp.nS, mdp.nA]) # REPLACE THIS LINE WITH YOUR CODE
    for state, actions in mdp.P.items():
        for action, reacts in actions.items():
            qpi = 0
            for react in reacts:
                prob = react[0]
                nextState = react[1]
                r = react[2]
                qpi += prob*(r+gamma*vpi[nextState])
            Qpi[state, action] = qpi    
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
    
#%% Policy Iteration
def policy_iteration(mdp, gamma, nIt, grade_print=print):
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.nS,dtype='int')
    pis.append(pi_prev)
    grade_print("Iteration | # chg actions | V[0]")
    grade_print("----------+---------------+---------")
    for it in range(nIt):
        # you need to compute qpi which is the state-action values for current pi
        # and compute the greedily policy, pi, from qpi
        vpi = compute_vpi(pi_prev, mdp, gamma)
        qpi = compute_qpi(vpi, mdp, gamma)
        pi = np.argmax(qpi, axis=1)

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
