# 模式识别编程作业，反向传播，Kmeans，谱聚类，svm调包libsvm

## 1. backpropagation.py

反向传播算法，单样本随机更新和批量更新。

## 2. kmeans/kmeans.py 

Kmeans聚类算法，Kmeans聚类。

## 3. spectral cluster.py

谱聚类算法，使用上述Kmeans做最后的聚类操作。使用了K临近的方法进行亲和度矩阵的计算，远距离的亲和度置为零。

## 4. svm/svm_classification.py, svm/visualization.py

svm进行minist手写体图像分类，首先标准化，之后进行十折交叉验证、网格法进行参数寻优。
