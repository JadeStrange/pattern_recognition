import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualization():
    data = np.loadtxt('a.csv', delimiter=',')


    # 这里是创建一个数据
    c_list = [0.01, 0.1, 0.5, 1, 5, 10, 100]
    g_list = [0.0008, 0.001, 0.0012, 0.0014, 0.0016, 0.0018, 0.0020]



    # 这里是创建一个画布
    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # 这里是修改标签
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(g_list)))
    ax.set_yticks(np.arange(len(c_list)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(g_list)
    ax.set_yticklabels(c_list)

    # 因为x轴的标签太长了，需要旋转一下，更加好看
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 添加每个热力块的具体数值
    # Loop over data dimensions and create text annotations.
    for i in range(len(c_list)):
        for j in range(len(g_list)):
            text = ax.text(j, i, data[i, j],
                           ha="center", va="center", color="w")
    ax.set_title("accuracy of 0/8 classification with change of g and c")
    fig.tight_layout()
    plt.colorbar(im)
    plt.show()

if __name__ == '__main__':
    visualization()