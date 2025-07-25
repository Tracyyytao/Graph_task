# 1. 节点分类
1.设置参数`--dataset citeseer --model gcn --visualize`后运行main.py文件，结果如下：

`Train / Val / Test Accuracy: [1.0, 0.64, 0.65]
 Total Training Time: 0.95 seconds`
 
![vis](https://github.com/Tracyyytao/Graph_task/blob/main/node_classification/assets/vis.png?raw=true)

2.设置参数`--dataset citeseer --model gcn --visualize --batch `进行邻居采样，结果如下：
`Train / Val / Test Accuracy: [1.0, 0.668, 0.672]
Total Training Time: 4.87 seconds`
