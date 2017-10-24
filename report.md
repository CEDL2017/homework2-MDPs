# Homework1 report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned. <br/>
這次的作業主要是用來熟悉這些工具怎麼使用 <br/>
以及用一些簡單的例子(測資) 去實作出value iteration, policy iteration.. 等等, <br/>
<br/>
實作的細節上，需要花些時間去了解一開始資料是怎麼給的，以及我應該如何妥善地進行運算 <br/>

1. value iteration

```python
	for it in range(nIt):
		for s in range(mdp.nS):            
			a_array=np.zeros(mdp.nA)
            
        for a in range(mdp.nA): 
                mysum=0
                for ss in range(len(mdp.P[s][a])):
                    prob=mdp.P[s][a][ss][0]
                    nextstate=mdp.P[s][a][ss][1]
                    reward=mdp.P[s][a][ss][2]

                    mysum=mysum+prob*(reward+GAMMA*Vprev[nextstate])
                a_array[a]=mysum
            
		pi[s] = np.argmax(a_array)
		V[s] = a_array[max_a]
```
> 基於bellman-equation的精神，不斷地迭代它，讓它可以收斂到bellman-equation的解<br>
> max|V-Vprev|越來越小，上課有談到這個方法的會收斂(if gamma in (0,1))
<br>
<br>

2a. state value function
```python
		for s in range(mdp.nS):
			a[s][s] = 1; #I
        for p in mdp.P[s][pi[s]]:
            prob=p[0]
            ss=p[1] #s'
            reward=p[2]
            
            b[s] = b[s] + prob * reward  #P*R => b
            a[s][ss] = a[s][ss] - gamma * prob #I-\gamma*P
    V = np.linalg.solve(a, b)    
```
> 這邊在解一個linear equation，利用的是V後來會到達到一個穩定狀態(stationary state)<br>
> 因此做一個線性求解(反函數)，去得到V<br>

2b. state-action value function
```python
    Qpi = np.zeros([mdp.nS, mdp.nA]) # Initial
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            my_sum=0
            for pi in mdp.P[s][a]:
                prob=pi[0]
                nextstate=pi[1]
                reward=pi[2]
                my_sum = my_sum + prob * (reward+gamma*vpi[nextstate])
            Qpi[s][a]=my_sum   
```
> 計算Q-value<br>

3. Sampling-based Tabular Q-Learning
> 利用random.random() 去模擬random sampling的過程<br>

## 心得
1. 這次作業在視覺化呈現的部分，原本助教附的圖表+步驟化圖示對學習相當地有幫助!<br>
2. 要先了解mdp.P裡面的資料儲存方式，並且重新對這些式子理解，才能有效地寫完這份作業<br>
3. 感覺RL很有趣，在下次作業應該還會繼續學習它<br>
