# Homework2 report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.

###
其實就是照 Value iteration 、 policy iteration 和 Q-Learning 的公式去打，然後程式就會動了
唯一值得 mention 的地方是 Problem 2a 要用 closed form 解 State-Value function 的部分
我原本的作法是：
###
	P = np.zeros((mdp.nS, mdp.nS))
	R = np.zeros(mdp.nS)
	for s in range(mdp.nS):
		for p in mdp.P[s][pi[s]]:
			P[s][p[1]] += p[0]
        		R[s] += p[2]
	a = np.eye(mdp.nS) - gamma * P
	b = P.dot(R)
	V = np.linalg.solve(a, b)
###
但發現這樣算不準，會有數值上的問題。
所以之後改成這樣：

###
	a = np.eye(mdp.nS)
	v = np.zeros(mdp.nS)
	for s in range(mdp.nS):
		for p in mdp.P[s][pi[s]]:
			a[s][p[1]] -= p[0]*gamma
        		b[s] += p[0] * p[2]
	V = np.linalg.solve(a, b)
###

感覺用 iteration method 還是比較好，一方面可以避免算反矩陣，一方面也可以避免這種難以debug 的數值問題。
