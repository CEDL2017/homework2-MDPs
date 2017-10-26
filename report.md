# Homework1 report
  
<hr />

前兩部分使用的是FrozenLakeEnv的環境模組  
要操控Agent在4*4的結冰水池上找到飛盤  
如果撿到飛盤得到+1結束  
如果中途掉進水裡-1結束  
## Value Iteration  
此方法的計算使用到馬可夫決策過程  
但在計算上各狀態的value計算相依  
所以迭帶步長逐步看時間延長到無限(足夠大)時
該位置的最大獲利以及該移動方向為何  
  
結果：  
![VI](https://github.com/w95wayne10/homework2-MDPs/blob/master/imgs/VI.png)

## Policy Iteration
此方法類似Value Iteration  
但是由state value function 和 state-action value function交替迭帶  
在代入policy到state value function 解 linear system後  
得到各狀態的值  
再將值引入state-action value function  
如value iteration一般但更新各個動作的期望分數  
選擇最大值成為新的policy繼續迴圈  
  
結果：  
![PI](https://github.com/w95wayne10/homework2-MDPs/blob/master/imgs/PI.png)
  
<hr />

第三部分使用的是CrawlingRobotEnv的環境模組  
要操控Agent在9*13的頭和爪子的呼應情況  
找到一個方式前進  
## Q-Learning  
由於情況不易枚舉或是前後環境變動問題  
而不能採用原先兩種方法  
Q-Learning便是採取sampling-based的方式解決問題
程式部分分三段  
    
第一段是設計下一步的走法  
設定一個門檻值  
小於這個值便隨機移動  
大於這個值便往最大獲利方向移動  
  
第二段設計Q-learning value的update方式  
每走到下一步時  
下一步所能獲得的最大值影響到這狀態走這步的分數  

第三段則是把兩個動作合併  
成為一個完整的學習過程  

結果：  
訓練約85000次迴圈後皆可達到3.37的平均速率