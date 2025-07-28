# 1. 节点分类
1.设置参数`--dataset citeseer --model gcn --visualize`后运行main.py文件，结果如下：

`Train / Val / Test Accuracy: [1.0, 0.64, 0.65]
 Total Training Time: 0.95 seconds`
 
![vis](https://github.com/Tracyyytao/Graph_task/blob/main/node_classification/assets/vis.png?raw=true)

2.设置参数`--dataset citeseer --model gcn --visualize --batch `进行邻居采样，结果如下：

`Train / Val / Test Accuracy: [1.0, 0.668, 0.672]
Total Training Time: 4.87 seconds`

# 2.  图上的链路预测
1.设置参数`--model GIN --dataset Cora`，运行main.py文件，结果如下：

![full](https://github.com/Tracyyytao/Graph_task/blob/main/link_prediction/assets/full.png?raw=true)

# 3.  图分类
以TuDataset 中的PROTEINS数据集为例，不同模型结合不同的池化方法，结果如下：

1.GCN

![GCN](https://github.com/Tracyyytao/Graph_task/blob/main/graph_classification/assets/GCN.png?raw=true)

2.GAT

![GAT](https://github.com/Tracyyytao/Graph_task/blob/main/graph_classification/assets/GAT.png?raw=true)

3.GraphSAGE

![GraphSAGE](https://github.com/Tracyyytao/Graph_task/blob/main/graph_classification/assets/GraphSAGE.png?raw=true)

4.GIN

![GIN](https://github.com/Tracyyytao/Graph_task/blob/main/graph_classification/assets/GIN.png?raw=true)

# 4.  知识图谱
运行期间日志如下：

![kg](https://github.com/Tracyyytao/Graph_task/blob/main/kg/assets/kg.png?raw=true)
