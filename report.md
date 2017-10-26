# Homework2 report

TA: try to elaborate the algorithms that you implemented and any details worth mentioned.

這次作業是應用MDP在類似DP的遊戲裡面，
前半部分在value iteration和policy iteration的部分，
只要照著hint和仔細照著上面的公式做基本上就沒什麼問題了。

下半部分要對Crawlingbot做Q-Learning的練習，
我還是有去找Q-Learning的相關資料來參考，
像是去搜尋epsilon greedy的具體實踐，
而在Q-Learning update那部分也是照著公式做，
最後將MDP實踐於environment內這次作業就算完成了。

比較頭痛的是關於policy iteration的部分有數值上的小問題。
答案要求的是0.00000，而我的實作出來的是-0.00000。
是非常小幾可忽略的誤差，我認為可能的原因在於比較low-level的計算問題，
因為前面在處理Vpi, Qpi的test都是pass的，所以可以合理地認為演算法實現是對的。

