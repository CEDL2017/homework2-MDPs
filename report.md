# Homework1 report

## Value Iteration:
照著 Value Iteration 的公式進行實作：
![](https://github.com/hellochick/homework2-MDPs/blob/master/imgs/value_iteration.png)

首先我利用兩個 for-loop 去遍歷每個 state，接著就可以去看所以可能的 action，透過 mdp.P 的結構去得到每個可能的 action 的機率、能獲得的 reward，有了這些資訊後，我們便可以算出下一步可能得到的最大值去更新當前 state 的 value。

```python
for s in range(mdp.nS):
    action_v = np.zeros(mdp.nA)
    for a in range(mdp.nA):
      value = 0
      for prob, next_state, reward in mdp.P[s][a]:
          value += prob * (reward + gamma * Vprev[next_state])
                
      action_v[a] = value
                
    V[s] = np.max(action_v)
```
## Policy iteration:
在 Policy iteration 這大題，分成了兩部分，首先我們需要先求出 state-value function Vpi：
![](https://github.com/hellochick/homework2-MDPs/blob/master/imgs/state_value_function.png)

在這邊我是參考 David Silver 課程當中所提到的推導，跟助教給的 hint 稍微不一樣，不過整體概念是相同的，都是透過線性轉換、反矩陣來算出 Vpi：
![](https://github.com/hellochick/homework2-MDPs/blob/master/imgs/bellman_equation.png)
```python
for s in range(mdp.nS):
    a = pi[s]
    
    for prob, next_s, reward in mdp.P[s][a]:
        P[s][next_s] += prob
        R[s] += prob * reward

    I = np.identity(mdp.nS)
    a = I - gamma * P
    b = R
    V = np.linalg.solve(a,b)
```
在算出 Vpi 後，我們便可以透過 Vpi 來求出 Qpi，也就是所謂的 state-action value function，這兩者的差異最簡單來看便是 vpi 表達的是每一個 state 的 value，而 Qpi 表達的是每一個 state 的每個 action 所代表的 value，在這邊同樣利用公式進行實作：  

![](https://github.com/hellochick/homework2-MDPs/blob/master/imgs/state_action_value_function.png)

```python
Qpi = np.zeros([mdp.nS, mdp.nA]) 
for s in range(mdp.nS):
    for a in range(mdp.nA):
        value = 0
        for prob, next_s, reward in mdp.P[s][a]:
            value += prob * (reward + gamma * vpi[next_s])
                
        Qpi[s][a] = value
```
算出這兩項 function 後便能進行 policy itearation，不過在這邊我出來的結果有一項為 -0.00000 與助教給的答案 +0.00000 並不相同，但我猜想也許是在線性運算的過程中有些誤差造成的，整體的觀念應該是正確的。

## Sampling-based Tabular Q-Learning
