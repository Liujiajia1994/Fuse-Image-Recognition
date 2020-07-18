import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
import mkl.kernel_tools as k_helpers
import mkl.multiple_kernel_implement as algo1

GLCM_file = 'F:/GitHub/Fuse-Image-Recognition/features/GLCM_16_features.txt'
FD_file = 'F:/GitHub/Fuse-Image-Recognition/features/FD_features.txt'
Harris_file = 'F:/GitHub/Fuse-Image-Recognition/features/Harris_4489_features.txt'
target_file = 'F:/download_SAR_data/experiment_data/dataset/target.txt'


def kernel_chose(feature, kernel_type, gamma):
    if kernel_type == 'linear':
        kernel_functions = [k_helpers.create_linear_kernel]
    elif kernel_type == 'rbf':
        kernel_functions = [k_helpers.create_rbf_kernel(gamma)]
    elif kernel_type == 'poly':
        kernel_functions = [k_helpers.create_poly_kernel(2, gamma)]
    elif kernel_type == 'exponential':
        kernel_functions = [k_helpers.create_exponential_kernel(gamma)]
    elif kernel_type == 'sigmoid':
        kernel_functions = [k_helpers.create_sigmoid_kernel(gamma)]
    else:
        kernel_functions = [k_helpers.create_histogram_kernel]
    kernel_matrices = k_helpers.get_all_kernels(feature, kernel_functions)
    return kernel_matrices


if __name__ == '__main__':
    # 分别获取三个特征的数据
    GLCM_data = np.genfromtxt(GLCM_file, dtype=str, delimiter=' ').astype(float)
    FD_data = np.genfromtxt(FD_file, dtype=str, delimiter=' ').astype(float)
    Harris_data = np.genfromtxt(Harris_file, dtype=str, delimiter=' ').astype(float)
    target = np.genfromtxt(target_file, dtype=str, delimiter=' ').astype(float)
    for j in range(target.size):
        if target[j] != 0:
            target[j] = -1
        else:
            target[j] = 1
    # 十折交叉验证
    num_k = 10
    skf = StratifiedKFold(n_splits=num_k)
    score_SVC = 0
    for train_index, test_index in skf.split(GLCM_data, target):
        GLCM_X_train, GLCM_X_test = GLCM_data[train_index], GLCM_data[test_index]
        FD_X_train, FD_X_test = FD_data[train_index], FD_data[test_index]
        Harris_X_train, Harris_X_test = Harris_data[train_index], Harris_data[test_index]
        y_train, y_test = target[train_index], target[test_index]
        # 设置gamma
        gamma = 1.0 / GLCM_X_train.shape[1]
        # gamma = 0.1
        kernel_train_matrics = []

        # 合成核矩阵
        GLCM_train_matrics = kernel_chose(GLCM_X_train, 'histogram', gamma)[0]
        FD_train_matrics = kernel_chose(FD_X_train, 'histogram', gamma)[0]
        Harris_train_matrics = kernel_chose(Harris_X_train, 'exponential', gamma)[0]

        kernel_train_matrics.append(GLCM_train_matrics)
        kernel_train_matrics.append(FD_train_matrics)
        kernel_train_matrics.append(Harris_train_matrics)

        C = 1
        init_weight = np.ones(3) / 3
        weights, final_train_data, J, alpha, duality_gap, final_gamma = algo1.find_kernel_weights(init_weight,
                                                                    kernel_train_matrics, C, y_train, 1, gamma)
        print('************************最后算到的结果************************')
        print('weights', weights)
        # print('combined_kernel', combined_kernel)
        # print('J', J)
        # print('alpha', alpha)
        # print('duality_gap', duality_gap)
        print('gamma', final_gamma)
        clf = svm.SVC(kernel='precomputed')
        clf.fit(final_train_data, y_train)

        # 对测试数据进行处理

        kernel_functions = [
            k_helpers.create_histogram_kernel,
            k_helpers.create_histogram_kernel,
            k_helpers.create_exponential_kernel(final_gamma),
        ]
        n_test = GLCM_X_test.shape[0]
        n_train = GLCM_X_train.shape[0]
        kernel_test_matrices = []
        GLCM_test_matrics = np.empty((n_test, n_train))
        FD_test_matrics = np.empty((n_test, n_train))
        Harris_test_matrics = np.empty((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                GLCM_test_matrics[i][j] = kernel_functions[0](GLCM_X_test[i], GLCM_X_train[j])
                FD_test_matrics[i][j] = kernel_functions[1](FD_X_test[i], FD_X_train[j])
                Harris_test_matrics[i][j] = kernel_functions[2](Harris_X_test[i], Harris_X_train[j])
        kernel_test_matrices.append(GLCM_test_matrics)
        kernel_test_matrices.append(FD_test_matrics)
        kernel_test_matrices.append(Harris_test_matrics)

        final_test_data = k_helpers.get_combined_kernel(kernel_test_matrices, weights)

        score_SVC += clf.score(final_test_data, y_test)
        print('一次循环的精度为%s' % (clf.score(final_test_data, y_test)))
    print('SVC最后的分类精度：%s' % (score_SVC / num_k))