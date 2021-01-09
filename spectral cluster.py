"""
谱聚类作业
"""


import numpy as np
import math

from sklearn import metrics

import kmeans.kmeans
import matplotlib.pyplot as plt
def loaddata():
    """
    加载数据
    :return:x_list数据
    """
    file_name = "2_data.txt"
    x_list = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            a_str = line.strip("\n")
            a_str = a_str.split(' ')
            x_list.append((float(a_str[0]), float(a_str[1])))
    return np.array(x_list)

def calculate_weight(x, y, sigma):
    """
    计算亲和度
    :param x: 两个向量
    :param y:
    :param sigma: 参数
    :return: 亲和度
    """
    return math.exp((-1) * np.linalg.norm(x - y, 2) / (2 * sigma * sigma))

def spectral_cluster(x_array):
    """
    谱聚类
    :param x_array: 数据集
    :return: 类别标签
    """
    weight = np.zeros((x_array.shape[0], x_array.shape[0]))

    for x_index, x in enumerate(x_array):
        for y_index, y in enumerate(x_array):
            weight[x_index][y_index] = calculate_weight(x, y, SIGMA)
        index_list = np.argsort(weight[x_index])
        for i in index_list[0: len(index_list) - K_NEIGHBOR]:
            weight[x_index][i] = 0
    weight = (weight + weight.T)/2
    D = np.zeros((x_array.shape[0], x_array.shape[0]))
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            D[i,i] = D[i,i] + weight[i, j]
    L = D - weight
    D_trans = np.zeros((x_array.shape[0], x_array.shape[0]))
    for i in range(D.shape[0]):
        D_trans[i,i] = 1/math.sqrt(D[i,i])
    L_sym = D_trans @ L @ D_trans

    # 计算特征值
    e_vals, e_vecs = np.linalg.eig(L_sym)

    sorted_indices = np.argsort(e_vals)
    # 过滤掉几乎为0的特征值对应的特征向量
    for i in range(len(sorted_indices)):
        if e_vals[sorted_indices[i]] <= 0.0000001:
            continue
        else:
            break
    print(i)
    V = e_vals[sorted_indices[i: K+i]]
    U = e_vecs[:, sorted_indices[i: K+i]]

    for i in range(U.shape[0]):
        norm = np.linalg.norm(U[i], 2)
        for j in range(U.shape[1]):
            U[i, j] = U[i, j]/norm
    predicts, cluster_center_point_list = kmeans.kmeans.kmeans(2, U)
    # np.savetxt('a.csv', predicts, fmt='%d', delimiter=',')  # 将数组a存为csv文件
    return predicts

if __name__ == '__main__':
    ground_truth = np.loadtxt('a.csv', delimiter=',')  # 将csv文件读取为数组
    x_array = loaddata()
    #SIGMA = 0.1 — 0.22大概
    #K = 2
    # 固定SIGMA为0.15 K从1——50
    SIGMA = 0.15
    K = 2
    acc_list = []
    for K_NEIGHBOR in range(4, 51, 1):
        predicts = spectral_cluster(x_array)
        acc_list.append(metrics.homogeneity_score(ground_truth, predicts))
    plt.figure()
    plt.plot(np.arange(4, 51, 1), np.array(acc_list))
    plt.title("fixed sigma=0.15, k is from 4 to 50")
    plt.xlabel("k")
    plt.ylabel("acc")
    plt.show()
    # 可视化聚类结果
    # plt.figure()
    # plt.scatter(x_array[:, 0], x_array[:, 1], c=predicts)
    # plt.show()

    K_NEIGHBOR = 10
    acc_list = []
    for SIGMA in np.arange(0.05, 0.31, 0.01):
        predicts = spectral_cluster(x_array)
        acc_list.append(metrics.homogeneity_score(ground_truth, predicts))
    plt.figure()
    plt.plot(np.arange(0.05, 0.31, 0.01), np.array(acc_list))
    plt.title("fixed k=10, sigma is from 0.05 to 0.3, with 0.01 step ")
    plt.xlabel("sigma")
    plt.ylabel("acc")
    plt.show()



