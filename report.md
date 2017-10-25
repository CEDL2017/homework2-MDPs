# 106061532洪彗庭 CEDL-hw2

In this assignment we have implemented value and policy iteration on MDP's dynamics model and also use sampling-based Q-Learning to learn to control a Crawler robot.

## Setup
Simply follow the instructions from TA, like follows:

```
conda env create -f environment.yml
source activate cedl
# deactivate when you want to leave the environment
source deactivate cedl
```

## Implementation
### Simple Gridworld - Frozen Lake environment
The environment looks like follows as described from gym: </br>

```   	 
        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
```

The MDP of this problem is specified as follows:

- _States(S)_:

```
        SFFF			0  1  2  3 
        FHFH			4  5  6  7
        FFFH     -> 8  9  10 11
        HFFG			12 13 14 15
```

- _Gamma(γ)_: 0.95 </br>
- _Actions(A)_:

```
0 : Left
1 : Down
2 : Right
3 : Top
```

- _Transition-of-states (P), Rewards (R)_:

```
========== in state 0 ==========
state 0, action 0, (prob, next-s, rw) = (0.1, 0, 0.0)
state 0, action 0, (prob, next-s, rw) = (0.8, 0, 0.0)
state 0, action 0, (prob, next-s, rw) = (0.1, 4, 0.0)
state 0, action 1, (prob, next-s, rw) = (0.1, 0, 0.0)
state 0, action 1, (prob, next-s, rw) = (0.8, 4, 0.0)
state 0, action 1, (prob, next-s, rw) = (0.1, 1, 0.0)
state 0, action 2, (prob, next-s, rw) = (0.1, 4, 0.0)
state 0, action 2, (prob, next-s, rw) = (0.8, 1, 0.0)
state 0, action 2, (prob, next-s, rw) = (0.1, 0, 0.0)
state 0, action 3, (prob, next-s, rw) = (0.1, 1, 0.0)
state 0, action 3, (prob, next-s, rw) = (0.8, 0, 0.0)
state 0, action 3, (prob, next-s, rw) = (0.1, 0, 0.0)
...
...
...
========== in state 15 ==========
state 15, action 0, (prob, next-s, rw) = (1.0, 15, 0)
state 15, action 1, (prob, next-s, rw) = (1.0, 15, 0)
state 15, action 2, (prob, next-s, rw) = (1.0, 15, 0)
state 15, action 3, (prob, next-s, rw) = (1.0, 15, 0)
```

here it saves all these information in a 2-level dictionary, **mdp.P**, to represent the states, possible actions in each state, and their rewards.

#### Value iteration
The concept in value iteration is base on Bellman Optimization Equation: </br>
![BOE](https://latex.codecogs.com/gif.latex?v%5E*%28s%29%20%3D%20%5Cmax_%7Ba%5Cin%20A%7D%20%28R%5Ea_s%20&plus;%20%5Cgamma%20%5Csum_%7Bs%27%5Cin%20S%20%7DP%5Ea_%7Bss%27%7D%20v%5E*%28s%27%29%29)

#### Policy iteration
### Sampling-based Tabular Q-Learning