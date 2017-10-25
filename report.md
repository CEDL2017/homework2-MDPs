# Homework2 report
105062575 &nbsp; 何元通
## Introduction
&nbsp; &nbsp; &nbsp; &nbsp; For this homework, we will complete agents which perform reinforcement learning. In the first and second part of the work, we are asked to complete a maze game with the agent. What is different between the two is that the former should be completed by value iteration and the latter should be done by policy iteration. The two perform different algorithm but are both able to finish the task well. FInally, for the last part of this assignment, a crawling robot is asked to be completed with tabular Q-learning. We would like to train the agent crawling with the speed around 3.37 in sufficient iteration.

## Implementation
### Value Iteration
&nbsp; &nbsp; &nbsp; &nbsp; Since the assignment is almost completed, we simply need to finish the critical part of the algorithm. For the first part, the value iteration is adopted. Just follow the hint, with three loops, I complete this task. in the first loop, I walk through all grid in the maze. For this way, I would like to update the value table grid by grid. Thus, for each grid, I list all actions the agent may take in this corresponding grid. Then, compute the values of all actions in the third loop and eventually choose the max one as the new value. Perform these loop structure for several iteration and the value table will converge and show a relatively better path from any non-terminal grid to the goal grid.  
The value iteration pseudo code is shown as following:
```
for grid in maze
  for actions
    for s in possible next state
      append Ps*(Rs+gamma*Vs) to v_list
  V_grid <- max(v_list)
  pr_grid <- argmax(action, v_list)

```

### Policy Iteration
&nbsp; &nbsp; &nbsp; &nbsp; As to policy iteration, it is divied into three functions. The one is the function which compute the value of a taken policy. Also, following with the hint, we simply have to perform a linear computation. I compute a matrix, a, and a vector, b, in the hint for each grid. The matrix, a, approximately means the weighted probabilities that a state "not" transit to another state. And, the vector, b, approximately means the expected reward of this state. For the two component, we will be able to compute the vector V, which fulfill the linear equation aV=b.
```
for grid in maze
  for s in next possible state
    as <- as - gamma*Ps
    b <- b + Ps*Rs
  compute V for aV=b
```
The next function finds the q values for each action given v values.
```
for grid in maze
  for a in action
    q[grid][a] <- sum( Ps*(Rs + gamma*Vs) ) for s in possible next state
```
And, for the last function, we only call the last two function to get the q-value table and find the best action with maximum q-value for each grid. Run this step for several iteration, and you will get a nice policy to finish this maze.  

### Tabular Q Learning
&nbsp; &nbsp; &nbsp; &nbsp; For the last part of this assignment, we complete the three function uesd for a tabular q learning algorithm. The spsilon greedy function is done easily. I simply find a best action of this state and make it having a probability to not do so. It is like a noise in action selection. For the q value updating function, I just compute the two equations of target computing and q updating. Since almost arguments have been provided, I only need to compute the possible action it may take in the next state before computing the two equations.Eventually, the last one performs q learning algorithm. With the two functions above, I merely need to arrange them into an appropriate place and record the next state. I first take a possible action and its corresponding reward and next state for current state. Then, updating the q-value table and record the next state as current state. Run it for sevaral iteration till it converge. The final speed will converge to about 3.37.

## Discussion
### Value Iteration & Policy Iteration
&nbsp; &nbsp; &nbsp; &nbsp; To my opinion, I cohnsider the value iteration and policy iteration work similarly. They search all of the grid and try to find the best strategy. I consider the difference between them just the one makes decision based on acculative scores and the other does according to the action. Also, it is apparently that policy iteration converge quicklier than value iteration. I consider it may because the value of value iteration updates base on the next possible state and the value of policy iteration updates based on the all actions. It is like that the value iteration consider the local information of each grid. But, the policy iteration considers all the actions. It is like considering the global information. One action change will cause all the others. Consequently, contrary to gradually update grid by grid, updating all actions in one ieteration looks faster.  

<p><img src="./imgs/value iteration.JPG" height = "270" /> <img src="./imgs/policy iteration.JPG" height = "270" /><p/>

&nbsp; &nbsp; &nbsp; &nbsp; Furthermore, I have a problem in compute vpi function inference. From the hint, we compute V by aV=b. But, in this hint, it seems to assume V[s]=V[s'] so that we are able to compute V. However, I consider this assumption may not always true. Thus, I doubt the assumption to be for simpler computation.  

### Tabular Q Learning
&nbsp; &nbsp; &nbsp; &nbsp; Contrary to the other two part, tabular q learning does not require to access to dynamic model. It maintains a q value table to fulfill the similar task. To my aspect, it is similar to assume the future condition and find the best expeted value related to the current state. I consider it is more useful in reality. Moreover, as there is a randomness in action selection, this can prevent the agent trapped in local optima. Also, it works like try and error, I consider it more intellegnet; it will not stupidly go through all states and take a large computation cost, thought it may be able to always find the global optima. Eventually, I consider it a promising, useful, and interesting algorithm.
