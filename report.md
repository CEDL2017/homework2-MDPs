# CEDL2017 HW2 Report
**Strongly recommended to view this report on this [HackMD link](https://hackmd.io/s/Sk0S7skT-), since GitHub does not support math equaitons**

Author: [Howard Lo (羅右鈞)](https://www.facebook.com/yuchunlo1206)

Here are some extra notes about implementing this homework.

We'll first briefly explain the two exact methods for solving MDPs problem, which are *value iteration* and *policy iteration* algorithms. Also, we'll point out the difference of *Bellman equations* between the complete one in the book and simplified one in this homework settings. And finally, *tabular-Q learning*.

For value iteration and policy iteration, the main reference I took was the second edition of [*"Reinforcement Learning: An Introduction"* book by Richard S. Sutton](http://incompleteideas.net/sutton/book/bookdraft2017june.pdf), and also the [lecture 2 and 3 of RL course by David Silver](https://www.youtube.com/playlist?list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-). And for tabular Q-learing, I recommend to take a look at the [lecture 10 of AI course by Pieter Abbeel](https://www.youtube.com/watch?v=7huURSBATmg&list=PLIeooNSdhQE5kRrB71yu5yP9BRCJCSbMt&index=11) to grasp the idea.

## Value Iteration
Value iteration algorithm uses *Bellman optimality equation* to iteratively update the state-value function $v$ until it is converged to optimal $v_*$.
$$
\begin{aligned}
v_{k+1}(s)&\leftarrow \underset{a}{\max} \mathbb{E}[R_{t+1}+ \gamma v_k(S_{t+1})|S_t=s,A_t=a]\\
&\leftarrow\underset{a}{\max}\underset{s',r}{\sum}p(s',r|s,a)[r+ \gamma v_k(s')]\\
&\leftarrow\underset{a}{\max}q_k(s,a)
\end{aligned}
$$

According to the book, the whole value iteration algorithm in pseudocode is shown as follows:

![](https://i.imgur.com/tpf1Wh4.png)

The algorithm simply says that:
1. Use the old state-value function $v_k(s)$ to compute the new state-action value function $q_{k+1}(s,a)$ by the recurive relationship between $v$ and $q$.
2. Choose the maximum action-state value function $q_{k+1}(s,a)$ to be our new state-value function $v_{k+1}(s)$, and use that to repeat step 1~2 until we get the optimal $v_*$ and the optimal $q_*$, which means until they are converged.
2. Eventually, we can then select the action which maximizes the $q_*$ to be our final optimal policy $\pi_*$.

Here is what the algorithm is implemented in python code looks like (there are some comments which are related to the policy iteration algorithm, we'll see that later):
```python=
# This is just the inner loop in Value Iteration algorithm.

# Initialization
V = np.zeros(mdp.nS) # state value
Q = np.zeros((mdp.nS, mdp.nA)) # state-action value
pi = np.zeros(mdp.nS) # policy (action for every state)

# For every state `s`, compute v(s) = \max_{a} Q(s,a) (Q(s,a) = state-action values)
for state in range(mdp.nS):

    ######################################################################################
    # This block is actually the same as `compute_qpi()` below.
    # One step lookahead to compute state-action value
    # state-action value = expectation of (immediate reward + discount factor * value of next state `v`)
    for action in range(mdp.nA):
        possible_transitions = mdp.P[state][action]
        for prob, next_state, reward in possible_transitions:
            Q[state][action] += prob * (reward + gamma * Vprev[next_state])

    ######################################################################################

    # Select best action value
    # This is actually *Bellman optimality equation* for value function
    V[state] = np.max(Q[state])
    # Create a deterministic policy using the optimal state-action value
    pi[state] = np.argmax(Q[state])

# The process of value iteration can be actually view as policy iteration:
# `V[state] = np.max(Q[state])` => one sweep of truncated policy evaluation
# `compute_qpi()` + `pi[state] = np.argmax(Q[state])` => one sweep of policy improvement

# Commented by Howard Lo.
```

Note that there are some differences compared to the pseudocode:
1. In our homework settings, we only get a single deterministic reward after the agent performs an action, so we can ignore the $\underset{r}{\sum}$ term in the complete Bellman equation. As a result, we use the simplified Bellman equation to compute $q$:
$$q(s,a) = \underset{s'}{\sum}p(s'|s,a)[r+\gamma v(s')]$$
And rewrite it to the update rule will be:
$$q_{k+1}(s,a) \leftarrow \underset{s'}{\sum}p(s'|s,a)[r+\gamma v_k(s')]$$
So that's why the $q$ is computed by:
```python=
Q[state][action] += prob * (reward + gamma * Vprev[next_state])
```

2. Why we put `pi` inside the loop is that we want to visualize the difference between different `pi`s in every iteration. So of course, if you're not going to do the visualization, compute the final `pi` after the loop will be fine.

And the extra question I had thought of: Why not directly perform value iteration on $q$ value function? Here is [the answer from Quora](https://www.quora.com/What-are-the-advantages-of-using-Q-value-iteration-versus-value-iteration-in-reinforcement-learning):

<div style="width:100%;text-align:center">
<img src="https://i.imgur.com/Fvs48ti.png" width="70%"/>
</div>


## Policy Iteration
Policy iteration is the process of *policy evaluation* and *policy improvement*, which can be illustrated by the following figure:

![](https://i.imgur.com/KXDPULg.png)

Briefly speaking, we initially take a random policy $\pi$, then compute a state-value function $v_{\pi}$ and use $v_{\pi}$ to compute $q_{\pi}$. After that, we select the new *greedy* policy $\pi'(s)$ from $q_{\pi}$:
$$\pi'(s)=\underset{a}{\operatorname{argmax}}q_{\pi}(s,a)$$

which the *policy improvement theorem* (detailed proof is on the page 87 of the book) tells us that:
$$v_{\pi'}(s) \geq q_{\pi}(s, \pi'(s)) \geq v_{\pi}(s)$$

where $q_{\pi}(s, \pi'(s))$ means that for some state $s$, we would like to select a new action $a$ from the new greedy policy $\pi'(s)$ that looks best in the short term, and therefore follow the orignal policy $\pi$ all the time.

So, eventually, by repeating this process, we will finally get our optimal value function $v_*$ and optimal policy $\pi_*$.

### Policy Evaluation
Given a policy, you evaluate it by computing the state-value function based on fixed policy $v_{\pi}$ to know whether the policy is good or bad. Here are two ways to compute:

1. Use *Bellman expectation equation* to iterativley update the state-value function $v_{\pi}$:
$$
\begin{aligned}
v_{\pi}^{k+1}(s) &\leftarrow \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}^k(S_{t+1})|S_t=s]\\
&\leftarrow\underset{a}{\sum}\pi(a|s)\underset{s',r}{\sum}p(s',r|s,a)[r+\gamma v_{\pi}^k(s')]\\
&\leftarrow\underset{s'}{\sum}p(s'|s)[r+\gamma v_{\pi}^k(s')]\text{ (Induced MRP. See note below)}
\end{aligned}
$$
Note: The simplified equation ignores the $\sum_a$ term (then $\pi(a|s)$ becomes $1$) and the $\sum_r$ term since in our homework settings, both of the policy and the reward are deterministic. So, when we ignore other possible actions we might take in some state, we then induce MDP to MRP (*Markov Reward Process*).<br>
For stochastic policy, we do not ignore the $\sum_a$ term. You can refer to the more complete version of [policy evaluation code by Denny Britz](https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Evaluation%20Solution.ipynb), which takes account of the probability of actions we might take.

2. Directly solve linear equation (Bellman expectation equation) to get the exact state-value function $v_{\pi}$:
$$v_{\pi}(s)=\underset{s'}{\sum}p(s'|s)[r+\gamma v_{\pi}(s')]\text{ (This is same as the above update rule)}
$$
We rewrite to its matrix form:
$$
\begin{bmatrix}
    v(s=1)\\
    v(s=2)\\
    \vdots\\
    v(s=n)
\end{bmatrix} =
\begin{bmatrix}
    p(s'=1|s=1) & \ldots & p(s'=1|s=n)\\
    p(s'=2|s=1) & \ldots & p(s'=2|s=n)\\
    \vdots & \ddots & \vdots\\
    p(s'=n|s=1) & \ldots & p(s'=n|s=n)
\end{bmatrix}
\left(
    \begin{bmatrix}
        r(s'=1)\\
        r(s'=2)\\
        \vdots\\
        r(s'=n)
    \end{bmatrix} + \gamma
    \begin{bmatrix}
        v(s'=1)\\
        v(s'=2)\\
        \vdots\\
        v(s'=n)
    \end{bmatrix}
\right)
$$
And keep deriving this...
$$
\begin{align}
v &= P(r+\gamma v)\\
v &= Pr+\gamma Pv\\
v-\gamma Pv &= Pr\\
(I-\gamma P)v &= Pr\\
v &= (I-\gamma P)^{-1}Pr
\end{align}
$$
Finally, we can solve the exact $v$ in the second-last equation by `numpy.linalg.solve()` or directly compute $v$ in the last equation by `numpy.linalg.inv()` and `numpy.dot()` .

Note that we use method 2 in our homework in order to get rid of numerical error, but if we have larger state number, it may require iterative method (e.g. method 1) to solve.

Now, let's take a look at what the code looks like for solving linear equation (method 2):
```python=
def compute_vpi(pi, mdp, gamma):
    # Probability of state transition matrix based on fixed deterministic policy `pi`.
    P = np.zeros((mdp.nS, mdp.nS))
    # Expection of immediate reward of state based on fixed deterministic policy `pi`.
    R = np.zeros(mdp.nS)

    for state in range(mdp.nS):
        action = pi[state]
        possible_transitions = mdp.P[state][action]
        for prob, next_state, reward in possible_transitions:
            P[state][next_state] += prob # There are duplicate next_state in P[state][action]
            R[state] += prob * reward # Expection of immediate reward R

    # Reference: MDP with fixed policy can be induced to MRP, then we can directly solve linear equation.
    # https://www.cs.cmu.edu/~katef/DeepRLControlCourse/lectures/lecture3_mdp_planning.pdf
    V = np.linalg.solve(np.eye(len(P)) - gamma*P, R)

    # Commented by Howard Lo.
    return V
```

Here are some caveats when implementing method 2:
- In order to pass the test code in our homework, we use `numpy.linalg.solve()` instead of `numpy.linalg.inv()` due to the numerical error.
- The `R` in our code is actually the term $Pr$ in our equation.


### Policy Improvement
Now, we know the $v_{\pi}$, we can use it to compute $q_{\pi}$ by their recursive relationship:
$$q_{\pi}(s,a) = \underset{s'}{\sum}p(s'|s,a)[r+\gamma v_{\pi}(s')]$$
The code is very intuitive, too:
```python=
def compute_qpi(vpi, mdp, gamma):
    Qpi = np.zeros([mdp.nS, mdp.nA])

    for state in range(mdp.nS):
        for action in range(mdp.nA):
            possible_transitions = mdp.P[state][action]
            for prob, next_state, reward in possible_transitions:
                Qpi[state][action] += prob * (reward + gamma * vpi[next_state])
    return Qpi
```
Finally, we can improve our policy by:
$$\pi'(s)=\underset{a}{\operatorname{argmax}}q_{\pi}(s,a)$$
Correspond to the code:
```python=
pi = np.argmax(Qpi, axis=1)
```

To wrap up, the simplified version of policy iteration code is just like:
```python=
# Start by a random policy or fixed policy, whatever...
pi_prev = np.zeros(mdp.nS)
for _ in range(num_iteration):
    # Policy evaluation
    vpi = compute_vpi(pi_prev, mdp, gamma)
    # Policy improvement
    Qpi = compute_qpi(vpi, mdp, gamma)
    pi = np.argmax(Qpi, axis=1)

    # Converged if there are no changes between the new policy and the old policy.
    if (pi != pi_prev).sum() == 0: break
    # Update the new policy
    else:  pi_prev = pi
```

Note that either value iteration or policy iteration is sufficient to solve the MDPs problem, and actually it turns out that value iteration is just the truncated version of policy iteration, so don't be confused :-)

## Sampling-based Tabular Q-Learning
If now, we do not have the prior information about the environment dynamics, specifially, we do not know the state-transition probabilites $p(s'|s,a)$ and reward $r$, then how do we compute the value functions?

we can actually do sampling! That is, we perform *[temporal difference learning](https://en.wikipedia.org/wiki/Temporal_difference_learning)*, by doing the action $\pi(s)$, take sample of outcome $s',r$ and then perform the [moving average (specifially, the exponential moving average)](https://en.wikipedia.org/wiki/Moving_average) to compute the estimated new value functions. And yes, this is the same as the policy evaluation, but we estimate it. Let's recap how we perform policy evaluation on $v_{\pi}(s)$:
$$
v_{\pi}(s)=\underset{s'}{\sum}p(s'|s)[r+\gamma v_{\pi}(s')]
$$

And we can estimate $v_{\pi}$ without knowing the environment dynamics:
- Sample of $v_{\pi}^k$: $sample = r + \gamma v_{\pi}^k(s')$
- Update to $v_{\pi}^{k+1}$: $v_{\pi}^{k+1} \leftarrow (1-\alpha)v_{\pi}^k + (\alpha)sample$
- Or same as: $v_{\pi}^{k+1} \leftarrow v_{\pi}^k + \alpha(sample - v_{\pi}^k)$
- $\alpha$ can be thought as a *learning rate*, which is a hyperparameter we need to tune (the idea seems to be similar to the *[stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)*).

But, here comes a problem. As we saw previously in the policy improvement part, we improve our policy by argmaxing $q_{\pi}(s,a)$, but before that, we need to compute $q_{\pi}(s,a)$:
$$q_{\pi}(s,a) = \underset{s'}{\sum}p(s'|s,a)[r+\gamma v_{\pi}(s')]$$
According to the above equation, we still need to know the state-transition probabilites $p(s'|s,a)$ and reward $r$. So, why don't we directly perform temporal difference learning on $q_{\pi}$ value function? And this turns out to be *Q-learning*!

Q-learning -- sample-based Q-value iteration:
$$
q_{\pi}^{k+1}(s,a) \leftarrow \underset{s'}{\sum}p(s'|s,a)[r+\gamma \underset{a'}{\max}q_{\pi}^k(s',a')]
$$
And we estimate $q_{\pi}$ is just similar to how we estimate $v_{\pi}$:
- Sample of $q_{\pi}^k$: $r+\gamma \underset{a'}{\max}q_{\pi}^k(s',a')$
- Update to $q_{\pi}^{k+1}$: $q_{\pi}^{k+1} \leftarrow (1-\alpha)q_{\pi}^k + (\alpha)sample$

Finally, the Q-value update code looks like:
```python=
def q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward):
    target = reward + gamma*np.max(q_vals[next_state])
    q_vals[cur_state][action] = (1 - alpha)*q_vals[cur_state][action] + alpha*target
```
And put it all together to form Q-learning:
```python=
for _ in range(num_iteration):
    action = eps_greedy(q_vals, eps, cur_state)
    next_state, reward, done, info = env.step(action)
    q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward)
    cur_state = next_state
```
