# Homework2 report

陳則銘 105062576

## Introduction
In this lab, we will solve **Markov Decision Processes (MDPs) with finite state and action spaces** with several classic algorithms that you learnt in the class.

The experiments here will use the Frozen Lake environment, a simple gridworld MDP that is taken from `gym` and slightly modified for this assignment. In this MDP, the agent must navigate from the start state to the goal state on a 4x4 grid, with stochastic transitions.

## Problem 1: implement value iteration

The pseudocode we need to implement:
> Initialize <a href="https://www.codecogs.com/eqnedit.php?latex=V^{(0)}(s)=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V^{(0)}(s)=0" title="V^{(0)}(s)=0" /></a>, for all <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a>
> - For <a href="https://www.codecogs.com/eqnedit.php?latex=i&space;=&space;0,1,2,..." target="_blank"><img src="https://latex.codecogs.com/gif.latex?i&space;=&space;0,1,2,..." title="i = 0,1,2,..." /></a>
<a href="https://www.codecogs.com/eqnedit.php?latex=V^{(i&plus;1)}(s)=max_a\sum_{s'}P(s,a,s')[R(s,a,s')&plus;\gamma&space;V^{(i)}(s')]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V^{(i&plus;1)}(s)=max_a\sum_{s'}P(s,a,s')[R(s,a,s')&plus;\gamma&space;V^{(i)}(s')]" title="V^{(i+1)}(s)=max_a\sum_{s'}P(s,a,s')[R(s,a,s')+\gamma V^{(i)}(s')]" /></a>, for all <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a>
---------
We additionally define the sequence of greedy policies:
> <a href="https://www.codecogs.com/eqnedit.php?latex=\pi&space;^{(i)}(s)=argmax_a\sum_{s'}P(s,a,s')[R(s,a,s')&plus;\gamma&space;V^{(i)}(s')]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi&space;^{(i)}(s)=argmax_a\sum_{s'}P(s,a,s')[R(s,a,s')&plus;\gamma&space;V^{(i)}(s')]" title="\pi ^{(i)}(s)=argmax_a\sum_{s'}P(s,a,s')[R(s,a,s')+\gamma V^{(i)}(s')]" /></a>

Our code will return two lists:<a href="https://www.codecogs.com/eqnedit.php?latex=[V^{(0)},V^{(1)},...,V^{(n)}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?[V^{(0)},V^{(1)},...,V^{(n)}]" title="[V^{(0)},V^{(1)},...,V^{(n)}]" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=[\pi^{(0)},\pi^{(1)},...,\pi^{(n-1)}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?[\pi^{(0)},\pi^{(1)},...,\pi^{(n-1)}]" title="[\pi^{(0)},\pi^{(1)},...,\pi^{(n-1)}]" /></a>

```
for it in range(nIt):
    oldpi = pis[-1] if len(pis) > 0 else None 
    Vprev = Vs[-1]
    V = []
    pi = []
    for sta in range(mdp.nS):
        maxv = 0
        maxpi = 0
        for act in range(mdp.nA):
            v = 0
            for n in mdp.P[sta][act]:
                v = v + n[0]*( n[2] + gamma*Vprev[n[1]] )
            if (maxv < v):
                maxv = v
                maxpi = act

        V = np.append(V, maxv)
        pi = np.append(pi, maxpi)
     Vs.append(V)
     pis.append(pi)
```

![1](imgs/1.png "Visualize results:")
![2](imgs/2.png)
![3](imgs/3.png)
![4](imgs/4.png)
![5](imgs/5.png)
![6](imgs/6.png)
![7](imgs/7.png)
![8](imgs/8.png)
![9](imgs/9.png)
![10](imgs/10.png)

## Problem 2: Policy Iteration
---

## Problem 3: Sampling-based Tabular Q-Learning
---
