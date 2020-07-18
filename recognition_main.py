import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


target_file = 'F:/download_SAR_data/experiment_data/dataset/target.txt'
data_file = './features/GLCM_16_features.txt'
# data_file = './features/FD_features.txt'
# data_file = './features/Harris_4489_features.txt'
# data_file = './features/GLCM_16_FD+Harris_4489_features.txt'

if __name__ == '__main__':
    score_DT = score_SVC = score_KNN = score_MLP = 0
    score_linear_SVC = score_rbf_SVC = score_poly_SVC = 0
    data = np.genfromtxt(data_file, dtype=str, delimiter=' ').astype(float)
    target = np.genfromtxt(target_file, dtype=str, delimiter=',').astype(int)
    # 还是采用十折交叉验证来分类
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(data, target):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]
        # 处理成二分类
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

        # # DT分类
        # clf = DecisionTreeClassifier(random_state=0)
        # clf.fit(X_train, y_train)
        # print('DT分类精度：%s' % (clf.score(X_test, y_test)))
        # score_DT += clf.score(X_test, y_test)
        # linear SVC分类
        clf_linear = SVC(kernel='linear')
        clf_linear.fit(X_train, y_train)
        print('linear_SVC分类精度：%s' % (clf_linear.score(X_test, y_test)))
        score_linear_SVC += clf_linear.score(X_test, y_test)
        # rbf SVC分类
        # clf_rbf = SVC(kernel='rbf')
        # clf_rbf.fit(X_train, y_train)
        # print('rbf_SVC分类精度：%s' % (clf_rbf.score(X_test, y_test)))
        # score_rbf_SVC += clf_rbf.score(X_test, y_test)
        # rbf SVC分类
        # clf_poly = SVC(kernel='sigmoid')
        # clf_poly.fit(X_train, y_train)
        # print('sigmoid_SVC分类精度：%s' % (clf_poly.score(X_test, y_test)))
        # score_poly_SVC += clf_poly.score(X_test, y_test)
    # #     KNN分类
    #     clf5 = KNeighborsClassifier(n_neighbors=3)
    #     clf5.fit(X_train, y_train)
    #     print('KNN分类精度：%s' % (clf5.score(X_test, y_test)))
    #     score_KNN += clf5.score(X_test, y_test)
    #     MLP分类
    #     clf6 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    #     clf6.fit(X_train, y_train)
    #     print('MLP分类精度：%s' % (clf6.score(X_test, y_test)))
    #     score_MLP += clf6.score(X_test, y_test)
    # print('DT最后的分类精度：%s' % (score_DT/10))
    print('linear SVC最后的分类精度：%s' % (score_linear_SVC/10))
    # print('poly3 SVC最后的分类精度：%s' % (score_rbf_SVC / 10))
    # print('poly4 SVC最后的分类精度：%s' % (score_poly_SVC / 10))
    # print('KNN最后的分类精度：%s' % (score_KNN/10))
    # print('MLP最后的分类精度：%s' % (score_MLP/10))