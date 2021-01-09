
"""
kmeans作业
"""
import random

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn import metrics


def load_data():
    def sub_load_data(file_path):
        data_list = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                a_str = line.strip("\n")
                a_str = list(filter(None, [str_number for str_number in a_str.split(' ')]))
                x = [float(str_number) for str_number in a_str]
                data_list.append(x)
        return data_list

    data_list = []
    label_list = []
    for label, file_path in enumerate(
            ["../kmeans_data/x1.txt", "../kmeans_data/x2.txt", "../kmeans_data/x3.txt", "../kmeans_data/x4.txt", "../kmeans_data/x5.txt"]):
        temp_data_list = sub_load_data(file_path)
        data_list.extend(temp_data_list)
        label_list.extend([label for _ in range(len(temp_data_list))])
    data_matrix = np.array(data_list)
    label_list = np.array(label_list)
    return data_matrix, label_list


def kmeans(cluster_n, X):
    """
    kmeans
    :param cluster_n: 分类数量
    :param X: 特征
    :return:
    """
    # 加载数据
    temp_max = np.max(X, axis=0)
    temp_min = np.min(X, axis=0)
    # 初始化
    cluster_center_point_list = []
    for i in range(cluster_n):
        temp_point = []
        for d in range(X.shape[1]):
            temp_point.append(random.uniform(temp_min[d], temp_max[d]))
        cluster_center_point_list.append(np.array(temp_point))
    # 循环
    label_array = np.ones(X.shape[0], dtype=np.int) * (-1)

    epoch = 0
    while True:

        epoch += 1
        for i in range(X.shape[0]):
            norm2_list = np.array([np.linalg.norm(cluster_center_point_list[cluster_i] - X[i], 2)
                                   for cluster_i in range(cluster_n)])
            temp_cluster_id = norm2_list.argmin()
            label_array[i] = temp_cluster_id

        new_cluster_center_point_sum_list = [np.zeros(X.shape[1]) for _ in range(cluster_n)]
        new_cluster_center_point_num_list = [0 for _ in range(cluster_n)]
        for data_id, label_id in enumerate(label_array):
            new_cluster_center_point_sum_list[label_id] = new_cluster_center_point_sum_list[label_id] + X[
                data_id]
            new_cluster_center_point_num_list[label_id] += 1

        new_cluster_center_point_list = []
        for cluster_center_id, cluster_center_point in enumerate(new_cluster_center_point_sum_list):

            if new_cluster_center_point_num_list[cluster_center_id] != 0:
                cluster_center_point = cluster_center_point / new_cluster_center_point_num_list[cluster_center_id]
            new_cluster_center_point_list.append(cluster_center_point)

        cluster_point_equal = True
        for cluster_id, new_cluster_center_point in enumerate(new_cluster_center_point_list):
            for d in range(new_cluster_center_point.shape[0]):
                if new_cluster_center_point[d] != cluster_center_point_list[cluster_id][d]:
                    cluster_point_equal = False
                    break
            if cluster_point_equal == False:
                break
        if cluster_point_equal:
            break
        else:
            cluster_center_point_list = new_cluster_center_point_list
    print("epoch:", epoch)
    return label_array, cluster_center_point_list


if __name__ == '__main__':
    X, Y = load_data()
    predicts, cluster_center_point_list = kmeans(5, X)



    # 聚类准确率
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(Y, predicts))
    # 聚类中心和真值中心的距离
    ground_truth_average_list = []
    for label in range(5):
        temp_average = np.average(X[np.where(Y == label)], axis=0)
        ground_truth_average_list.append(temp_average)
    wucha = 0
    for ground_truth_average in ground_truth_average_list:
        norm2_list = [np.linalg.norm(ground_truth_average - cluster_center_point, 2) for cluster_center_point in
                      cluster_center_point_list]
        val = np.min(norm2_list)
        wucha += val
    wucha /= len(ground_truth_average_list)
    print("聚类中心", cluster_center_point_list)
    print("聚类中心与真值均值之间的误差: %0.3f" % wucha)

    # 展示点
    plt.figure("Kmeans results")
    plt.scatter(X[:, 0], X[:, 1], c=predicts)
    plt.show()