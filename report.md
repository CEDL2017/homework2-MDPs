# Homework2 report (105061523)

In this homework, we will solve Markov Decision Processes (MDPs) with finite state and action spaces with several classic algorithms that you learnt in the class.


## Problem 1: implement value iteration
### Methodology
Value iteration is a method of computing an optimal MDP policy and its value. Value iteration starts at the "end" and then works backward, refining an estimate of either Q or V. There is really no end, so it uses an arbitrary end point. Let Vk be the value function assuming there are k stages to go, and let Qk be the Q-function assuming there are k stages to go. These can be defined recursively.

### Pseudo Code
Initialize <a href="https://www.codecogs.com/eqnedit.php?latex=V^{(0)}(s)=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V^{(0)}(s)=0" title="V^{(0)}(s)=0" /></a>, for all <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a>

For <a href="https://www.codecogs.com/eqnedit.php?latex=i=0,&space;1,&space;2,&space;\dots" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i=0,&space;1,&space;2,&space;\dots" title="i=0, 1, 2, \dots" /></a>
- <a href="https://www.codecogs.com/eqnedit.php?latex=V^{(i&plus;1)}(s)&space;=&space;\max_a&space;\sum_{s'}&space;P(s,a,s')&space;[&space;R(s,a,s')&space;&plus;&space;\gamma&space;V^{(i)}(s')]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V^{(i&plus;1)}(s)&space;=&space;\max_a&space;\sum_{s'}&space;P(s,a,s')&space;[&space;R(s,a,s')&space;&plus;&space;\gamma&space;V^{(i)}(s')]" title="V^{(i+1)}(s) = \max_a \sum_{s'} P(s,a,s') [ R(s,a,s') + \gamma V^{(i)}(s')]" /></a>, for all <a href="https://www.codecogs.com/eqnedit.php?latex=s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s" title="s" /></a>

### Result
<table border=1>
<tr>
	<td>
	<img src="./imgs/valueIteration-0.png" width="19%"/>
	<img src="./imgs/valueIteration-1.png" width="19%"/>
	<img src="./imgs/valueIteration-2.png" width="19%"/>
	<img src="./imgs/valueIteration-3.png" width="19%"/>
	<img src="./imgs/valueIteration-4.png" width="19%"/>
	</td>
</tr>
<tr>
	<td>
	<img src="./imgs/valueIteration-5.png" width="19%"/>
	<img src="./imgs/valueIteration-6.png" width="19%"/>
	<img src="./imgs/valueIteration-7.png" width="19%"/>
	<img src="./imgs/valueIteration-8.png" width="19%"/>
	<img src="./imgs/valueIteration-9.png" width="19%"/>
	</td>
</tr>
</table>
<br>



## Problem 2: Policy Iteration


### Methodology
* Step a
	<p>
	Use function `compute_vpi` that computes the state-value function <a href="https://www.codecogs.com/eqnedit.php?latex=V^{\pi}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V^{\pi}" title="V^{\pi}" /></a> for an arbitrary policy <a href="https://www.codecogs.com/eqnedit.php?latex=\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi" title="\pi" /></a>.
	Recall that <a href="https://www.codecogs.com/eqnedit.php?latex=V^{\pi}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V^{\pi}" title="V^{\pi}" /></a> satisfies the following linear equation:
	<a href="https://www.codecogs.com/eqnedit.php?latex=V^{\pi}(s)&space;=&space;\sum_{s'}&space;P(s,\pi(s),s')[&space;R(s,\pi(s),s')&space;&plus;&space;\gamma&space;V^{\pi}(s')]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V^{\pi}(s)&space;=&space;\sum_{s'}&space;P(s,\pi(s),s')[&space;R(s,\pi(s),s')&space;&plus;&space;\gamma&space;V^{\pi}(s')]" title="V^{\pi}(s) = \sum_{s'} P(s,\pi(s),s')[ R(s,\pi(s),s') + \gamma V^{\pi}(s')]" /></a>
	</p>
* Step b
	<p>
	Write a function to compute the state-action value function <a href="https://www.codecogs.com/eqnedit.php?latex=Q^{\pi}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q^{\pi}" title="Q^{\pi}" /></a>, defined as follows <a href="https://www.codecogs.com/eqnedit.php?latex=Q^{\pi}(s,&space;a)&space;=&space;\sum_{s'}&space;P(s,a,s')[&space;R(s,a,s')&space;&plus;&space;\gamma&space;V^{\pi}(s')]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q^{\pi}(s,&space;a)&space;=&space;\sum_{s'}&space;P(s,a,s')[&space;R(s,a,s')&space;&plus;&space;\gamma&space;V^{\pi}(s')]" title="Q^{\pi}(s, a) = \sum_{s'} P(s,a,s')[ R(s,a,s') + \gamma V^{\pi}(s')]" /></a>
	</p>

### Pseudo Code
Initialize <a href="https://www.codecogs.com/eqnedit.php?latex=\pi_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi_0" title="\pi_0" /></a> <br>

For <a href="https://www.codecogs.com/eqnedit.php?latex=n=0,&space;1,&space;2,&space;\dots" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n=0,&space;1,&space;2,&space;\dots" title="n=0, 1, 2, \dots" /></a>
- Compute the state-value function <a href="https://www.codecogs.com/eqnedit.php?latex=V^{\pi_{n}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V^{\pi_{n}}" title="V^{\pi_{n}}" /></a>
- Using <a href="https://www.codecogs.com/eqnedit.php?latex=V^{\pi_{n}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V^{\pi_{n}}" title="V^{\pi_{n}}" /></a>, compute the state-action-value function <a href="https://www.codecogs.com/eqnedit.php?latex=Q^{\pi_{n}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q^{\pi_{n}}" title="Q^{\pi_{n}}" /></a>
- Compute new policy <a href="https://www.codecogs.com/eqnedit.php?latex=\pi_{n&plus;1}(s)&space;=&space;\operatorname*{argmax}_a&space;Q^{\pi_{n}}(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi_{n&plus;1}(s)&space;=&space;\operatorname*{argmax}_a&space;Q^{\pi_{n}}(s,a)" title="\pi_{n+1}(s) = \operatorname*{argmax}_a Q^{\pi_{n}}(s,a)" /></a>

### Result
<img src="./imgs/policy_iteration.png" width="50%"/>



## Problem 3: Sampling-based Tabular Q-Learning
<img src="./imgs/crawler.png" width="100%"/>

### Pseudo Code
Q learning update. After we observe a transition <a href="https://www.codecogs.com/eqnedit.php?latex=s,&space;a,&space;s',&space;r" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s,&space;a,&space;s',&space;r" title="s, a, s', r" /></a>, <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\textrm{target}(s')&space;=&space;R(s,a,s')&space;&plus;&space;\gamma&space;\max_{a'}&space;Q_{\theta_k}(s',a')" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textrm{target}(s')&space;=&space;R(s,a,s')&space;&plus;&space;\gamma&space;\max_{a'}&space;Q_{\theta_k}(s',a')" title="\textrm{target}(s') = R(s,a,s') + \gamma \max_{a'} Q_{\theta_k}(s',a')" /></a> <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=Q_{k&plus;1}(s,a)&space;\leftarrow&space;(1-\alpha)&space;Q_k(s,a)&space;&plus;&space;\alpha&space;\left[&space;\textrm{target}(s')&space;\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q_{k&plus;1}(s,a)&space;\leftarrow&space;(1-\alpha)&space;Q_k(s,a)&space;&plus;&space;\alpha&space;\left[&space;\textrm{target}(s')&space;\right]" title="Q_{k+1}(s,a) \leftarrow (1-\alpha) Q_k(s,a) + \alpha \left[ \textrm{target}(s') \right]" /></a>


### Result
<img src="./imgs/qlearning.png" width="50%"/>
