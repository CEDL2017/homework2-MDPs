# Homework2 Report

## Value Iteration

For value iteration, we'll estimate a value ![](https://render.githubusercontent.com/render/math?math=V%5E%7B%5Cpi%7D%28s%29%20%3D%20%5Csum_%7Bs%27%7D%20P%28s%2C%5Cpi%28s%29%2Cs%27%29%5B%20R%28s%2C%5Cpi%28s%29%2Cs%27%29%20%2B%20%5Cgamma%20V%5E%7B%5Cpi%7D%28s%27%29%5D&mode=display) for every state, and our policy is simply to select the action which can transit the state to another state with largest expected value, ![](https://render.githubusercontent.com/render/math?math=%5Cpi%5E%7B%28i%29%7D%28s%29%20%3D%20%5Carg%20%5Cmax_a%20%5Csum_%7Bs%27%7D%20P%28s%2Ca%2Cs%27%29%20%5B%20R%28s%2Ca%2Cs%27%29%20%2B%20%5Cgamma%20V%5E%7B%28i%29%7D%28s%27%29%5D&mode=display).

After several iterations, we can observe that the action taken at each state can obtain a higher expected value.

![](imgs/ValueItr.png)

## Policy Iteration

For policy iteration, we will need to compute the state-value function ![](https://render.githubusercontent.com/render/math?math=V%5E%7B%5Cpi%7D%28s%29%20%3D%20%5Csum_%7Bs%27%7D%20P%28s%2C%5Cpi%28s%29%2Cs%27%29%5B%20R%28s%2C%5Cpi%28s%29%2Cs%27%29%20%2B%20%5Cgamma%20V%5E%7B%5Cpi%7D%28s%27%29%5D&mode=display).

Differ from the value function, the **state-value function** returns sum of expected value of **all the possible next state** while value function only take the expected value of the next state which is brought by the optimal action into consideration.

In the implementation, the state-value function could be treated as simultaneous equations, and we use `np.linalg.solve` to obtain the exact solution for each state.

Once the state-value function is obtained, we further need the **state-action value function** to decide which action to take in a certain state. The state-action value function is defined as follow: ![](https://render.githubusercontent.com/render/math?math=Q%5E%7B%5Cpi%7D%28s%2C%20a%29%20%3D%20%5Csum_%7Bs%27%7D%20P%28s%2Ca%2Cs%27%29%5B%20R%28s%2Ca%2Cs%27%29%20%2B%20%5Cgamma%20V%5E%7B%5Cpi%7D%28s%27%29%5D&mode=display), which returns a representation of expected reward for a pair of given state and action.

Then we can model our policy as ![](https://render.githubusercontent.com/render/math?math=%5Cpi_%7Bn%2B1%7D%28s%29%20%3D%20%5Coperatorname%2A%7Bargmax%7D_a%20Q%5E%7B%5Cpi_%7Bn%7D%7D%28s%2Ca%29&mode=inline), which directly uses Q-value to decide the action.

The rewards of several states increase faster than they do in value iteration.

![](imgs/PolicyItr.png)

## Sampling-based Tabular Q-Learning

Instead of using the transition function and the state-value function to compute the Q-function, we can also directly establish a table where each cell represents one or more states to model Q-function. The Q-value of every possible action at state s is stored in the cell of state s. Same as before, the policy is to select action at state s which can return the highest Q-value.

For the CrawlingRobotEnv(), the reward converges to 3.37 after 1e5 iterations.

```
Itr 0 # Average speed: 0.05
Itr 50000 # Average speed: 2.03
Itr 100000 # Average speed: 3.37
Itr 150000 # Average speed: 3.37
Itr 200000 # Average speed: 3.37
Itr 250000 # Average speed: 3.37
```