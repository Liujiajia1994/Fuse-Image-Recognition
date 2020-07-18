"""
    这里主要是想用多核学习方法来融合实现特征融合，
    结合github上的代码，做了稍微的调整
    write by July Liu
    time: 2018/11/13
"""
# 这里先做个测试
import datetime

import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedKFold

import mkl.kernel_tools as k_helpers
import mkl.multiple_kernel_implement as algo1
from sklearn import preprocessing

GLCM_file = 'F:/GitHub/Fuse-Image-Recognition/features/GLCM_features.txt'
FD_file = 'F:/GitHub/Fuse-Image-Recognition/features/FD_features.txt'
Harris_file = 'F:/GitHub/Fuse-Image-Recognition/features/Harris_features.txt'
data_file = 'F:/GitHub/Fuse-Image-Recognition/features/GLCM+FD+Harris_features.txt'
target_file = 'F:/download_SAR_data/experiment_data/dataset/target.txt'


def seven_train_thirty_test(data_file, target_file):
    # 导入数据
    X_data = np.genfromtxt(data_file, dtype=str, delimiter=' ').astype(float)
    y_target = np.genfromtxt(target_file, dtype=str, delimiter=' ').astype(int)
#     取出前70%的数据作为训练集，30%的数据作为测试集
    X_train = X_data[0:int(X_data.shape[0]*0.7), :]
    y_train = y_target[0:int(y_target.size*0.7)]
    X_test = X_data[int(X_data.shape[0]*0.7):, :]
    y_test = y_target[int(y_target.size * 0.7):]
    return X_train, y_train, X_test, y_test


# 4个不同核函数的SVM分类器
def classifier(train_data, train_labels, test_data, test_labels):
    # linear，poly, rbf, sigmoid, precomputed（自定义核函数）
    clf_linear = svm.SVC(kernel='linear')
    clf_linear.fit(train_data, train_labels)
    print('linear分类后精度为%s' % (clf_linear.score(test_data, test_labels)))

    clf_poly = svm.SVC(kernel='poly')
    clf_poly.fit(train_data, train_labels)
    print('poly分类后精度为%s' % (clf_poly.score(test_data, test_labels)))

    clf_rbf = svm.SVC(kernel='rbf')
    clf_rbf.fit(train_data, train_labels)
    print('rbf分类后精度为%s' % (clf_rbf.score(test_data, test_labels)))
    #
    # clf_sigmoid = svm.SVC(kernel='sigmoid')
    # clf_sigmoid.fit(train_data, train_labels)
    # print('sigmoid分类后精度为%s' % (clf_sigmoid.score(test_data, test_labels)))


def train_kernel(train_data, train_labels):
    n, d = train_data.shape
    nlabels = train_labels.size

    # Make labels -1 and 1
    for i in range(nlabels):
        if train_labels[i] != 0:
            train_labels[i] = -1
        else:
            train_labels[i] = 1

    gamma = 1.0 / d

    kernel_functions = [
        k_helpers.create_linear_kernel,
        # k_helpers.create_rbf_kernel(gamma),
        # k_helpers.create_poly_kernel(2, gamma),
        # k_helpers.create_poly_kernel(3, gamma),
        # k_helpers.create_poly_kernel(4, gamma),
        # k_helpers.create_sigmoid_kernel(gamma),
    ]

    M = len(kernel_functions)

    # The weights of each kernel
    # Initialized to 1/M
    d = np.ones(M) / M
    # Stores all the individual kernel matrices
    kernel_matrices = k_helpers.get_all_kernels(train_data, kernel_functions)
    # 惩罚数值 C:penalty value => 0<=alpha_i<=C
    C = 1
    weights, combined_kernel, J, alpha, duality_gap, gamma = algo1.find_kernel_weights(d, kernel_matrices, C, train_labels, 1, gamma)
    print('************************最后算到的结果************************')
    print('weights', weights)
    print('combined_kernel', combined_kernel)
    print('J', J)
    print('alpha', alpha)
    print('duality_gap', duality_gap)
    print('gamma', gamma)
    return weights


if __name__ == '__main__':
    # # 70%作为训练集，30%作为测试集
    #
    # # GLCM+FD+Harris
    # train_data, train_data_labels, test_data, test_data_labels = seven_train_thirty_test(GLCM_file, target_file)
    # # train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    # start = datetime.datetime.now()
    # weights = train_kernel(train_data, train_data_labels)
    # end_all = datetime.datetime.now()
    # print('GLCM+FD+Harris训练的时间为', (end_all - start).seconds / 60)
    # print(weights)
    # # classifier(train_data, train_data_labels, test_data, test_data_labels)
    # 采用十折交叉验证
    data = np.genfromtxt(data_file, dtype=str, delimiter=' ').astype(float)
    target = np.genfromtxt(target_file, dtype=str, delimiter=',').astype(int)
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(data, target):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]





