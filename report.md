# Homework1 report
106062623 楊朝勛

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.

1. value iteration

```python
for it in range(nIt):
        oldpi = pis[-1] if len(pis) > 0 else None # \pi^{(it)} = Greedy[V^{(it-1)}]. Just used for printout
        Vprev = Vs[-1] # V^{(it)}

        Q = np.asarray([np.asarray([np.asarray([p*(r+gamma*Vprev[s]) for (p,s,r) in tups]).sum() \
                                  for (a, tups) in a2d.items()]) for (s, a2d) in mdp.P.items()]) #update Q value
        pi = np.argmax(Q, axis=1) #pi^{(it)} = Greedy[V^{(it)}]
        
        V = np.max(Q, axis=1) #V^{(it+1)} = T[V^{(it)}]
    return Vs, pis
```
透過更新reward value的值來更新下個iteration每個state的value <br>
其中每次選擇時都選擇`最大value`的state當下個輸入<br>
(i.e. 皆假設state和action space不隨時間變化)
<br>
2. policy iteration

```python
#Vpi值依據公式推導
  pi_tups = [a2d[pi[s]] for (s, a2d) in mdp.P.items()]
  A = np.identity(mdp.nS)
  transition = np.array([np.array([p * A[s2] for (p,s2,r) in tups]).sum(axis=0) for tups in pi_tups])
  A -= gamma * transition
  b = np.array([np.array([p * r for (p,s2,r) in tups]).sum() for tups in pi_tups])
  Vpi = np.linalg.solve(A, b)
#Qpi值依據公式推導 
  Qpi = np.array([np.array([np.array([p*(r+gamma*vpi[s2]) for (p,s2,r) in tups]).sum() for (a, tups) in a2d.items()]) for (s, a2d) in mdp.P.items]) 
    
```
透過更新reward value的值來更新下個iteration每個state的value <br>
其中每次選擇時根據`policy iteration`的方式去依據每個state之`機率`sample結果<br>
(i.e. 皆假設state和action space不隨時間變化)<br>
<br>

3. tabular Q-learning

```python
while(loop):
    action = eps_greedy(q_vals, eps, cur_state)
    next_state,reward, _, _ = env.step(action)
    q_learning_update(gamma, alpha, q_vals, cur_state, action, next_state, reward)
    cur_state = next_state  
```
透過更新reward value的值來更新下個iteration每個state的value <br>
會先隨機出一個大小`0~1`的數字<br>
當小於`epi`時,其中每次選擇時根據Q value tabel選擇最大value的state當下個輸入<br>
當大於`epi`時,其中每次選擇時隨機選擇state當下個輸入<br>
(i.e. 皆假設state和action space不隨時間變化)

