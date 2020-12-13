import random

import numpy as np
import matplotlib.pyplot as plt
import math

from mpl_toolkits.mplot3d import Axes3D

epoch_limit = 10000  # 轮数
CRITERION = 0.00001  # norm小于该值则停止
LEARNING_RATE = 0.02  # 学习率
INPUT_SIZE = 3  # 输入层节点数
HIDDEN_SIZE = 16  # 隐含层节点数
OUTPUT_SIZE = 3  # 输出层节点数
# 以上为可以调节的参数


weight_i_h = np.random.uniform(-1, 1, (INPUT_SIZE, HIDDEN_SIZE))
weight_h_j = np.random.uniform(-1, 1, (HIDDEN_SIZE, OUTPUT_SIZE))

input_point_array = np.zeros(INPUT_SIZE)
hidden_point_array = np.zeros(HIDDEN_SIZE)
output_point_array = np.zeros(OUTPUT_SIZE)


def forward_propagation(X_array):
    """
    前向跑网络
    :param X_array:一个样本
    :return:
    """
    global hidden_point_array, input_point_array, output_point_array, weight_i_h, weight_h_j
    input_point_array = X_array  # 输入层被赋值
    hidden_point_array = input_point_array @ weight_i_h  # 隐层用输入层矩阵乘权重
    hidden_point_array = tanhx(hidden_point_array)  # 隐层过一个双曲正切激励函数

    output_point_array = hidden_point_array @ weight_h_j  # 输出层用权重乘一下
    output_point_array = sigmoid(output_point_array)  # 过一个sigmoid激励函数

    return output_point_array


def delta_hidden_output_weight(target_value_array, output_value_array, f_derivative_array, y_array):
    """
    计算隐层到输出层的权重变化
    :param target_value_array: 目标值，长度为输出节点的个数
    :param output_value_array: 输出值，长度为输出节点的个数
    :param f_derivative_array: 激活函数的导数，长度为输出节点的个数
    :param y_array: 隐含层的输出，长度为隐含层节点的个数
    :return: delta_hidden_output_weight_matrix
    """
    delta_j_array = target_value_array - output_value_array  # 差值向量
    delta_j_array_after_derivative = f_derivative_array * delta_j_array  # 经过导数放缩后的向量

    return LEARNING_RATE * y_array.reshape((-1, 1)) @ delta_j_array_after_derivative.reshape(
        (-1, 1)).T  # 列向量 @ 行向量出来一个delta矩阵


def delta_input_hidden_weight(target_value_array,
                              output_value_array,
                              f_output_derivative_array,
                              f_hidden_derivative_array,
                              x_array):
    """
    计算输入层到隐层的权重变化
    :param target_value_array: 目标值，长度为输出节点的个数
    :param output_value_array: 输出值，长度为输出节点的个数
    :param f_output_derivative_array: 输出层激活函数的导数，长度为输出节点的个数
    :param f_hidden_derivative_array: 隐含层的激活函数导数向量， 长度为隐含层节点个数
    :param x_array: 输入向量，长度为输入节点的个数
    :return: delta_input_hidden_weight_matrix
    """
    delta_j_array = target_value_array - output_value_array  # 差值向量
    delta_j_array_after_derivative = f_output_derivative_array * delta_j_array  # 经过导数放缩后的向量

    delta_h_array = delta_j_array_after_derivative @ weight_h_j.T
    delta_h_array_after_derivative = f_hidden_derivative_array * delta_h_array

    return LEARNING_RATE * x_array.reshape((-1, 1)) @ delta_h_array_after_derivative.reshape((-1, 1)).T
    # 列向量 @ 行向量出来一个delta矩阵


def calculate_2_norm_of_gradient_for_all_samples(X_matrix, Y_matrix):
    """
    计算所有样本下的梯度，作为推出条件
    :param X_matrix:
    :param Y_matrix:
    :return:
    """
    w2_sum = np.zeros((HIDDEN_SIZE, OUTPUT_SIZE))
    w1_sum = np.zeros((INPUT_SIZE, HIDDEN_SIZE))
    for sample_id in range(X_matrix.shape[0]):
        output_value_array = forward_propagation(X_matrix[sample_id])  # 前传跑网络
        target_value_array = Y_matrix[sample_id]
        f_derivative_array = sigmoid_derivative(hidden_point_array @ weight_h_j)
        y_array = hidden_point_array
        delta_hidden_output_weight_matrix = delta_hidden_output_weight(target_value_array,
                                                                       output_value_array,
                                                                       f_derivative_array,
                                                                       y_array)

        f_hidden_derivative_array = tanhx_derivative(input_point_array @ weight_i_h)
        delta_input_hidden_weight_matrix = delta_input_hidden_weight(target_value_array=target_value_array,
                                                                     output_value_array=output_value_array,
                                                                     f_output_derivative_array=f_derivative_array,
                                                                     f_hidden_derivative_array=f_hidden_derivative_array,
                                                                     x_array=input_point_array)
        w2_matrix = delta_hidden_output_weight_matrix / ((-1) * LEARNING_RATE)
        w1_matrix = delta_input_hidden_weight_matrix / ((-1) * LEARNING_RATE)
        w2_sum = w2_sum + w2_matrix
        w1_sum = w1_sum + w1_matrix

    return math.sqrt(np.trace(w2_sum @ w2_sum.T) + np.trace(w1_sum @ w1_sum.T))


def piliang(X_matrix, Y_matrix):
    """
    Batch Backpropagation
    :return: w
    """
    global weight_i_h, weight_h_j
    epoch_now = 1
    loss_list = []
    while epoch_now <= epoch_limit:
        delta_weight_h_j_sum = 0
        delta_weight_i_h_sum = 0
        for sample_id in range(0, X_matrix.shape[0]):
            forward_propagation(X_matrix[sample_id])  # 前传跑网络
            target_value_array = Y_matrix[sample_id]
            output_value_array = output_point_array
            f_derivative_array = sigmoid_derivative(hidden_point_array @ weight_h_j)
            y_array = hidden_point_array
            delta_hidden_output_weight_matrix = delta_hidden_output_weight(target_value_array,
                                                                           output_value_array,
                                                                           f_derivative_array,
                                                                           y_array)

            f_hidden_derivative_array = tanhx_derivative(input_point_array @ weight_i_h)
            delta_input_hidden_weight_matrix = delta_input_hidden_weight(target_value_array=target_value_array,
                                                                         output_value_array=output_value_array,
                                                                         f_output_derivative_array=f_derivative_array,
                                                                         f_hidden_derivative_array=f_hidden_derivative_array,
                                                                         x_array=input_point_array)

            delta_weight_h_j_sum = delta_weight_h_j_sum + delta_hidden_output_weight_matrix
            delta_weight_i_h_sum = delta_weight_i_h_sum + delta_input_hidden_weight_matrix

        weight_i_h = weight_i_h + delta_weight_i_h_sum
        weight_h_j = weight_h_j + delta_weight_h_j_sum

        norm = calculate_2_norm_of_gradient_for_all_samples(X_matrix, Y_matrix)

        # loss
        z_list = []
        for sample_id in range(X_matrix.shape[0]):
            sample_X = X_matrix[sample_id]
            z_list.append(forward_propagation(sample_X))
        z_matrix = np.array(z_list)
        loss = squared_error_loss_function(Y_matrix, z_matrix)

        loss_list.append(loss)
        acc = accuracy(t_matrix=Y_matrix, z_matrix=z_matrix)
        print("epoch: ", epoch_now, ",norm: ", norm, ",loss: ", loss, ", acc: ", acc)
        epoch_now += 1
        if norm < CRITERION:
            break

    plt.figure()
    plt.plot(np.arange(len(loss_list)), np.array(loss_list))
    plt.title("batch updating")
    plt.show()


def accuracy(t_matrix, z_matrix):
    """计算两个矩阵之间的准确率"""
    true_num = 0
    for sample_id in range(t_matrix.shape[0]):

        temp_t = t_matrix[sample_id]
        temp_z = z_matrix[sample_id]

        max_t_id = -1
        max_t_id_value = -1000
        for temp_id in range(len(temp_t)):
            if temp_t[temp_id] > max_t_id_value:
                max_t_id = temp_id
                max_t_id_value = temp_t[temp_id]
        max_z_id = -1
        max_z_id_value = -1000
        for temp_id in range(len(temp_z)):
            if temp_z[temp_id] > max_z_id_value:
                max_z_id = temp_id
                max_z_id_value = temp_z[temp_id]

        if max_t_id == max_z_id:
            true_num += 1

    return true_num / t_matrix.shape[0]


def danyangben(X_matrix, Y_matrix):
    """
    Stochastic Backpropagation
    随机更新
    :return:
    """
    global weight_i_h, weight_h_j
    epoch_now = 1
    loss_list = []
    while epoch_now <= epoch_limit:
        sample_id = random.randint(0, X_matrix.shape[0] - 1)
        output_value_array = forward_propagation(X_matrix[sample_id])  # 前传跑网络
        target_value_array = Y_matrix[sample_id]  # groundTruth
        f_derivative_array = sigmoid_derivative(hidden_point_array @ weight_h_j)  # 求f'(netj)
        y_array = hidden_point_array  # 隐层值
        delta_hidden_output_weight_matrix = delta_hidden_output_weight(target_value_array=target_value_array,
                                                                       output_value_array=output_value_array,
                                                                       f_derivative_array=f_derivative_array,
                                                                       y_array=y_array)  # 算隐含层到输出层

        f_hidden_derivative_array = tanhx_derivative(input_point_array @ weight_i_h)  # f'2(neth)
        delta_input_hidden_weight_matrix = delta_input_hidden_weight(target_value_array=target_value_array,
                                                                     output_value_array=output_value_array,
                                                                     f_output_derivative_array=f_derivative_array,
                                                                     f_hidden_derivative_array=f_hidden_derivative_array,
                                                                     x_array=input_point_array)

        weight_h_j = weight_h_j + delta_hidden_output_weight_matrix
        weight_i_h = weight_i_h + delta_input_hidden_weight_matrix
        norm = calculate_2_norm_of_gradient_for_all_samples(X_matrix, Y_matrix)

        # loss
        z_list = []
        for sample_id in range(X_matrix.shape[0]):
            sample_X = X_matrix[sample_id]
            z_list.append(forward_propagation(sample_X))
        z_matrix = np.array(z_list)
        loss = squared_error_loss_function(Y_matrix, z_matrix)
        loss_list.append(loss)
        acc = accuracy(t_matrix=Y_matrix, z_matrix=z_matrix)
        print("epoch: ", epoch_now, ",norm: ", norm, ",loss: ", loss, ", acc: ", acc)
        epoch_now += 1
        if norm < CRITERION:
            break
    plt.figure()
    plt.plot(np.arange(len(loss_list)), np.array(loss_list))
    plt.title("stochastic updating")
    plt.show()


def sigmoid(x):
    """sigmoid激励函数"""
    return 1 / (1 + np.exp((-1) * x))


def sigmoid_derivative(x):
    """sigmoid激活函数的导数"""
    y = sigmoid(x)
    return y * (1 - y)


def tanhx(x):
    """双曲正切激励函数"""
    return (np.exp(x) - np.exp((-1) * x)) / (np.exp(x) + np.exp((-1) * x))


def tanhx_derivative(x):
    """双曲正切激励函数的导数(thx) ' = 1/(chx)^2
    chx = (e^x + e^(-x)/2"""
    chx = (np.exp(x) + np.exp((-1) * x)) / 2
    return 1 / (chx * chx)


def squared_error_loss_function(t_matrix_d_j, z_matrix_d_j):
    """
    平方误差损失函数
    :param t_matrix_d_j: 每行是一个样本，每一列是输出值
    :param z_matrix_d_j: 每行是一个样本，每一列是目标值
    :return: se_lossfunction
    """
    chazhi = t_matrix_d_j - z_matrix_d_j
    return (1 / 2) * np.trace((chazhi @ np.transpose(chazhi)))


def load_data():
    """
    加载数据
    :return: X_matrix, Y_matrix
    """
    x_list = []
    y_list = []
    f = open("data.txt", 'r')
    for line in f.readlines():
        a_str = line.strip("\n")
        x = [float(str_number) for str_number in a_str.split(' ')[0:3]]
        y = int(a_str[-1])
        x_list.append(x)
        y_list.append(y)
    print("输出x_list", x_list)
    print("输出y_list", y_list)
    X_matrix = np.array(x_list)
    y_array_list = []
    for y in y_list:
        y_array = np.zeros(OUTPUT_SIZE)
        y_array[y] = 1
        y_array_list.append(y_array)
    Y_matrix = np.array(y_array_list)

    return X_matrix, Y_matrix


if __name__ == '__main__':
    np.set_printoptions(suppress=True)  # 取消科学计数法输出
    X_matrix, Y_matrix = load_data()

    label_x_dic = {}
    for i in range(Y_matrix.shape[0]):
        if tuple(Y_matrix[i]) not in label_x_dic.keys():
            label_x_dic[tuple(Y_matrix[i])] = []
        label_x_dic[tuple(Y_matrix[i])].append(X_matrix[i])

    to_show_list = []
    for key in label_x_dic.keys():
        temp_array = np.array(label_x_dic[key])
        x = temp_array[:, 0]
        y = temp_array[:, 1]
        z = temp_array[:, 2]
        to_show_list.append([x, y, z])

    danyangben(X_matrix, Y_matrix)
    piliang(X_matrix, Y_matrix)
