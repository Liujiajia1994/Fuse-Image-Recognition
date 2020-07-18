import numpy as np
from numpy import *
from numpy import linalg as la

# 相似度计算
data_file = './FD_feature.txt'


def pearson_similar(a, b):
    # print(a, len(a))
    if len(a) < 3:
        return 1.0
    else:
        return 0.5+0.5*corrcoef(a, b, rowvar=0)[0][1]


def ecludSim(a, b):
    return 1.0/(1.0+la.norm(a-b))


if __name__ == '__main__':
    # 从文件中获取特征向量数据
    data = np.genfromtxt(data_file, dtype=str, delimiter=' ', usecols=range(280)).astype(float)
    # print("pearson_similar=", pearson_similar(data[0, :], data[2, :]))
    similar_list = []
    sum = temp =0
    for i in range(len(data)):
        # print("第1行与第%s行的ecludSim为%s" % (i+1,ecludSim(data[0, :], data[i, :])))
        similarity = ecludSim(data[0, :], data[i, :])
        similar_list.append(similarity)
    print(similar_list)
    # 求这个数组的平均值作为阈值
    ave = average(similar_list)
    # 大大于阈值设置+1，再除于数组的个数
    for k in range(len(similar_list)):
        if similar_list[k] > ave:
            temp += 1
    weight_feature = temp / len(similar_list)
    print('平均值为', weight_feature)