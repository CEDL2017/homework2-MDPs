# Homework1 report

本次的作業主要是利用 Bellman Equation 來實現三種不同的算法：
## Value iteration
  使用 bellman 的最優方程式來更新 value, 最後收斂得到的 value即是當前 state狀態下最佳的 value. 因此只要最後收斂，就可以利用最佳的值來推導最佳的 policy.
### Spotlight code
  ```
   for it in range(nIt):
        oldpi = pis[-1] if len(pis) > 0 else None # \pi^{(it)} = Greedy[V^{(it-1)}]. Just used for printout
        Vprev = Vs[-1] # V^{(it)}
        V = np.zeros([mdp.nS])
        pi = np.zeros([mdp.nS])
        for s in range(mdp.nS):
            A = np.zeros([mdp.nA])
            for a in range(mdp.nA):
                for  probability, nextstate, reward in mdp.P[s][a]:
                    #print(nextstate)
                    A[a] += probability*(reward+gamma*Vs[it][nextstate])
                       
            V[s] = np.max(A)
            pi[s] = np.argmax(A)
```    
## Policy iteration
  使用 bellman equation 來更新 value, 最後收斂的 value 即是當下 policy的 value值，目的是為了使後面的 ploicy improvement來得到新的 policy. 接者再進行下一次的迭代求下一個 value和最佳的 policy.
### Spotlight code
State value function
```
for state1 in range(mdp.nS):
        for prob, nextstate, reward in mdp.P[state1][pi[state1]]:
            
            a[state1][nextstate] -= gamma*prob
            b[state1] += prob*reward
        

    V = np.linalg.solve(a,b)
```
State policy(action value) function
```
Qpi = np.zeros([mdp.nS, mdp.nA]) # REPLACE THIS LINE WITH YOUR CODE
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for prob, nextstate, reward in mdp.P[s][a]:
                Qpi[s][a] += prob*(reward+gamma*vpi[nextstate])
```
## Tabular Q-Learning
  此方法用在不知道 current state to next state的機率，所以不能用 Bellman equation update參數，還有一個參數 eps來決定下一個動作是隨機動作還是取最佳動作
### Spotlight code
Eps greedy
```
    ran = random.random()
    if ran<eps:
        action = random.randint(0,len(q_vals[state])-1)
    else:
        action = np.argmax(q_vals[state])
```        
Q learning update
```
 target = max(q_vals[next_state][0], q_vals[next_state][1], q_vals[next_state][2], q_vals[next_state][3])
    q_vals[cur_state][action]= (1-alpha)*q_vals[cur_state][action]+alpha*(reward+gamma*target)
```    
