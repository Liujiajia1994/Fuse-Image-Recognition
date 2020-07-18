import numpy as np
import mkl.kernel_tools as k_helpers
from sklearn import svm, preprocessing
from mkl.multiKernelFeatureFusion import seven_train_thirty_test


weight_train = [0.03595884, 0.68635391, 0.27768725]


# 根据训练出来的核函数权重和核函数进行线性组合
def my_kernel(u):
    kernel_matrices = k_helpers.get_all_kernels(u, kernel_functions)
    combined_kernel_matrix = k_helpers.get_combined_kernel(kernel_matrices, weight_train)
    return combined_kernel_matrix


# 测试集进行处理
def test_kernel_processing(train_data, test_data, kernel_functions):
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


if __name__ == '__main__':
    data_file = 'F:/GitHub/Fuse-Image-Recognition/features/GLCM+FD+Harris_features.txt'
    target_file = 'F:/download_SAR_data/experiment_data/dataset/target.txt'
    train_data, train_labels, test_data, test_labels = seven_train_thirty_test(data_file, target_file)
    # train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    # test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    n, d = train_data.shape
    # 这里的gamma好像也是可以训练出来的
    # gamma = 1.0 / d
    gamma = 0.141081
    kernel_functions = [
        k_helpers.create_linear_kernel,
        k_helpers.create_rbf_kernel(gamma),
        k_helpers.create_poly_kernel(2, gamma),
    ]
    for j in range(train_labels.size):
        if train_labels[j] != 0:
            train_labels[j] = -1
        else:
            train_labels[j] = 1
    for i in range(test_labels.size):
        if test_labels[i] != 0:
            test_labels[i] = -1
        else:
            test_labels[i] = 1
    # 核函数与权重之间进行线性组合 形成新的合成后的核函数 给SVM 进行分类
    clf = svm.SVC(kernel='precomputed')
    new_train = my_kernel(train_data)
    clf.fit(new_train, train_labels)

    # 这里的测试集需要重新调整一下
    final_test_data = test_kernel_processing(train_data, test_data, kernel_functions)
    print('SVM分类后精度为%s' % (clf.score(final_test_data, test_labels)))