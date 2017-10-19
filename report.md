# Homework1 report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.

### Problem 1: implement value iteration
V = np.copy(Vprev) <- 如同hint所言，V要用Vprev的copy，不然就不會是題目要的更新方式。
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
