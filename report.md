# 105061516 Homework2 report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.


1. value iteration

```python
while(loop):
  v = value_update(v, env)
```

> 假設state之間的關係不會變化<br>
> 將reward-value table持續更新，讓每個state採取最新reward-value最大化的行動。
<br>
<br>
2. policy iteration

```python
while(loop):
  v = compute_vpi(pi, env)
  qpi = compute_qpi(v, env)
  pi = compute_pi(qpi)
```
> 假設state之間的關係不會變化<br>
> 將q-value table持續更新，讓其採取reward-value最大化的policy。
<br>
<br>
3. tabular Q-learning

```python
while(loop):
  action = get_best_act(qval, env)
  feedback = do_action(action, env)
  qval = q_update(qval, feedback)
```

> 將q-value table持續更新，讓其在有action-noise的狀況下採取reward-value最大化的policy。
