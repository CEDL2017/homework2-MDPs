# Homework1 report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned. <br/>
這次的作業主要是用來熟悉這些工具怎麼使用 <br/>
以及用一些簡單的例子(測資) 去實作出value iteration, policy iteration.. 等等, <br/>
<br/>
實作的細節上，需要花些時間去了解一開始資料是怎麼給的，以及我應該如何妥善地進行運算 <br/>

1. value iteration
```python
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
            
            max_a = np.argmax(a_array)
            max_sum = a_array[max_a]
            V[s]=max_sum
            pi[s]=max_a 
```  
