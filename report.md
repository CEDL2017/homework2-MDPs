# Homework1 report of 林杰(102000039)

#### Step 0: Unique

Since the example consist of several transitions which has same next_state while the current_state and action are the same. It causes some inconvinience, so I implemented a small functionality that sums up the probability of transitions has same (next_state, current_state, action).

#### Step 1: Value Iteration

Implement the greedy value interation update process for the environment, and return current best policy.

#### Step 2: Policy Iteration

This stop consist of 2 parts, "state-value iteration" and "state action iteration". This is a process that jumping back-and-forth between two greedy update iterations. And eventually converges to an optimal solution.

I faced a problem while implementing ```compute_vpi()``` function. I'm still not sure why this fails, but mathematically, I think I can dot the 'P' (probability) and 'R' (reward) right before solving linear system. Unfortunately, my output is different (with some errors, up to 0.1 in some states) to the solution given. But this error disappears when I turn back to multiply P and R inside for-loops. It is quite weird to me and I'll try to check out the reason afterward (when I have time).

#### Step 3: Sampling-based Tabular Q-Learning

This sounds hard, but basically very easy. Just randomly sample a random action. And use previously implemented functionality update values respectively. Then we can see the crawler crawls through the play ground very fast. To slower its motion, I added an 100 ms delay in each loop.

#### Conclusion

Although the Q-learning sounds very easy in class, but I faced several difficulty while implementing such intuitive algorithm. By implementing the details, I can properly justify whether I understand its concept and which part of it should I study again. This is really a good experiance, looking forward to next assignment and implement Q-learning in my personal project some day.
