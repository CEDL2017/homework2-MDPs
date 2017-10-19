# Homework1 report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.

基本上照著提供的公式跟提示打就可以完成了。不過有些地方是當時寫的時候覺得比較能講的︰

#### Problem 1: implement value iteration
```V = np.copy(Vprev)``` <- 如同hint所言，V要用Vprev的copy，不然就不會是題目要的更新方式。
```
def value_iteration(mdp, gamma, nIt, grade_print=print):
.
.
    for it in range(nIt):
          oldpi = pis[-1] if len(pis) > 0 else None
          Vprev = Vs[-1] # V^{(it)}
          V = np.copy(Vprev)
.
.
```
#### Problem 2a: state value function
```
def compute_vpi(pi, mdp, gamma):
.
.
    for s in range(mdp.nS):
        action = pi[s] 
        expect = 0.0
        trans_list = mdp.P[s][action]
        for trans_idx in range(len(trans_list)):
            p, next_s, r = trans_list[trans_idx]
            expect += p * r
            a[s, next_s] += gamma * p 
        b[s] = expect
    I = np.eye(mdp.nS)
    a = I - a
.
.
```
原本```a[s, next_s] += gamma * p```這行我是用```=```，不過後來發現當s到next_s有兩種以上的reward時會錯。
