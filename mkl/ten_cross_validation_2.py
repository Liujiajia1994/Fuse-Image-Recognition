# 用十折交叉验证 来测试一下多核学习的精度
import numpy as np
from sklearn import svm, preprocessing
from sklearn.model_selection import StratifiedKFold
import mkl.kernel_tools as k_helpers
import mkl.multiple_kernel_implement as algo1

weight_train = [1]


# 根据训练出来的核函数权重和核函数进行线性组合
# def my_kernel(u):
#     kernel_matrices = k_helpers.get_all_kernels(u, kernel_functions)
#     combined_kernel_matrix = k_helpers.get_combined_kernel(kernel_matrices, weight_train)
#     return combined_kernel_matrix
    # return kernel_matrices


# 测试集进行处理
def test_kernel_processing(train_data, test_data, kernel_functions, weight_train):
    n_test = test_data.shape[0]
    n_train = train_data.shape[0]
    M = len(kernel_functions)
    kernel_matrices = []
    for m in range(M):
        kernel_func = kernel_functions[m]
        kernel_matrices.append(np.empty((n_test, n_train)))

        # Creates kernel matrix
        for i in range(n_test):
            for j in range(n_train):
                kernel_matrices[m][i, j] = kernel_func(test_data[i], train_data[j])
    final_test_data = k_helpers.get_combined_kernel(kernel_matrices, weight_train)
    return final_test_data


def classiy(data, target):
    num_k = 10
    score_SVC = 0
    skf = StratifiedKFold(n_splits=num_k)
    for train_index, test_index in skf.split(data, target):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]
        n, d = X_train.shape
        # 这里的gamma好像也是可以训练出来的
        gamma = 1.0 / d
        kernel_functions = [
            # k_helpers.create_linear_kernel,
            # k_helpers.create_rbf_kernel(gamma),
            # k_helpers.create_poly_kernel(2, gamma),
            # k_helpers.create_exponential_kernel(gamma),
            k_helpers.create_histogram_kernel,
            # k_helpers.create_exponential_kernel(gamma),
            # k_helpers.create_sigmoid_kernel(gamma)
        ]
        for j in range(y_train.size):
            if y_train[j] != 0:
                y_train[j] = -1
            else:
                y_train[j] = 1
        for i in range(y_test.size):
            if y_test[i] != 0:
                y_test[i] = -1
            else:
                y_test[i] = 1
        # 核函数与权重之间进行线性组合 形成新的合成后的核函数 给SVM 进行分类
        clf = svm.SVC(kernel='precomputed')
        kernel_matrices = k_helpers.get_all_kernels(X_train, kernel_functions)
        # 固定权重设置
        new_train = k_helpers.get_combined_kernel(kernel_matrices, weight_train)
        clf.fit(new_train, y_train)
        # 训练权重
        # 惩罚数值 C:penalty value => 0<=alpha_i<=C
        # C = 1
        # M = len(kernel_functions)
        # pointD = np.ones(M) / M
        # weights, combined_kernel, J, alpha, duality_gap, final_gamma = algo1.find_kernel_weights(pointD, kernel_matrices, C, y_train, 1, gamma)
        # print('************************最后算到的结果************************')
        # print('weights', weights)
        # # print('combined_kernel', combined_kernel)
        # # print('J', J)
        # # print('alpha', alpha)
        # # print('duality_gap', duality_gap)
        # print('gamma', final_gamma)
        # # 训练出权重了以后，接下来就是合成新的矩阵
        # clf.fit(combined_kernel, y_train)

        # 这里的测试集需要重新调整一下
        # final_test_data = test_kernel_processing(X_train, X_test, kernel_functions, weights)
        # no train
        final_test_data = test_kernel_processing(X_train, X_test, kernel_functions, weight_train)

        score_SVC += clf.score(final_test_data, y_test)
        print('一次循环的精度为%s' % (clf.score(final_test_data, y_test)))
    print('SVC最后的分类精度：%s' % (score_SVC / num_k))


def random_data(data_x, data_y):
    indices = np.random.permutation(data_x.shape[0])
    data = data_x[indices]
    target = data_y[indices]
    return data, target


target_file = 'F:/download_SAR_data/experiment_data/dataset/target.txt'
data_file = 'F:/GitHub/Fuse-Image-Recognition/features/GLCM_FD_features.txt'
if __name__ == '__main__':
    data = np.genfromtxt(data_file, dtype=str, delimiter=' ').astype(float)
    target = np.genfromtxt(target_file, dtype=str, delimiter=',').astype(int)
    # 做个归一化 查看一下结果
    # data_norm = preprocessing.scale(data)
    # random_data(data, target)
    classiy(data, target)