from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from adaboost_classifier import AdaCostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import BernoulliRBM


def weight_set(samples):
    weight_samples = [1.0 / len(samples) for sample in samples]
    return weight_samples


def ten_fold(data_file, target_file):
    clfDTlist = adaBoostList = 0
    skf = StratifiedKFold(n_splits=10)
    X_data = np.genfromtxt(data_file, dtype=str, delimiter=' ', usecols=range(280)).astype(float)
    y_target = np.genfromtxt(target_file, dtype=str, delimiter=' ', usecols=range(1)).astype(int)
    for train_index, test_index in skf.split(X_data, y_target):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_target[train_index], y_target[test_index]
        # 权重设置
        weights = weight_set(X_train)
        # print(weights)
        # 对三个特征的特征向量进行分类
        ABC = AdaBoostClassifier(DecisionTreeClassifier(random_state=0), algorithm='SAMME.R')
        ABC.fit(X_train, y_train, weights)
        adaBoostList += ABC.score(X_test, y_test)
        print('*****')
        # print(ABC.estimator_weights_)
    # print(clfDTlist / 10)
    print(adaBoostList / 10)


def seven_train_thirty_test(data_file, target_file, val):
    # 导入数据
    X_data = np.genfromtxt(data_file, dtype=str, delimiter=' ', usecols=range(val)).astype(float)
    y_target = np.genfromtxt(target_file, dtype=str, delimiter=' ', usecols=range(1)).astype(int)
#     取出前70%的数据作为训练集，30%的数据作为测试集
    X_train = X_data[0:int(X_data.shape[0]*0.7), :]
    y_train = y_target[0:int(y_target.size*0.7)]
    X_test = X_data[int(X_data.shape[0]*0.7):, :]
    y_test = y_target[int(y_target.size * 0.7):]
    return X_train, y_train, X_test, y_test


def fuze_sample_weight(FD_path, GLCM_path, Harris_path):
    """
    融合这个权重比值
    :return: 返回融合后的权重矩阵1*(280+4+280)
    """
    FD_weight = np.genfromtxt(FD_path, dtype=str, delimiter=' ', usecols=range(280)).astype(float)
    GLCM_weight = np.genfromtxt(GLCM_path, dtype=str, delimiter=' ', usecols=range(4)).astype(float)
    Harris_weight = np.genfromtxt(Harris_path, dtype=str, delimiter=' ', usecols=range(280)).astype(float)
    new_weight = np.zeros(len(FD_weight)+len(GLCM_weight)+len(Harris_weight), dtype=float)
    # for i in range(len(FD_weight)):
    #     print(FD_weight[i]/3)
    for i in range(len(FD_weight)+len(GLCM_weight)+len(Harris_weight)):
        if i < 280:
            new_weight[i] = FD_weight[i]/3
        elif (i >= 280) and (i < 284):
            new_weight[i] = GLCM_weight[i-280]/3
        else:
            new_weight[i] = Harris_weight[i-284]/3
    return new_weight


def get_importance(clf, feature_importance_file):
    feature_importance = clf.feature_importances_
    # 将获取的特征重要性值存放入文件
    with open(feature_importance_file, 'a') as file:
        for i in range(len(feature_importance)):
            # print(feature_importance[i])
            file.write(str(feature_importance[i]) + ' ')
        file.close()


def adaboost_classifier(X_train, y_train, X_test, y_test):
    # 权重设置
    weights = weight_set(X_train)
    clf2 = AdaCostClassifier(DecisionTreeClassifier(random_state=0), algorithm='SAMME')
    clf2.fit(X_train, y_train, weights)
    score = clf2.score(X_test, y_test)
    return score


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = seven_train_thirty_test('./features/GLCM_HOG_pca_feature.txt',
                                                               './features/target', 40)
    # # 将权重赋给训练集，形成新的训练集
    # weight_list = fuze_sample_weight('./features/FD_importance.txt', './features/GLCM_importance.txt',
    #                                  './features/Harris_importance.txt')
    # new_train = np.zeros(X_train.shape, dtype=float)
    # for i in range(X_train.shape[0]):
    #     for j in range(X_train.shape[1]):
    #         new_train[i][j] = X_train[i][j]*weight_list[j]
    # print(new_train.shape)
    # # 测试集重新计算
    # new_test = np.zeros(X_test.shape, dtype=float)
    # for i in range(X_test.shape[0]):
    #     for j in range(X_test.shape[1]):
    #         new_test[i][j] = X_test[i][j]*weight_list[j]
    # 接下来进行训练并测试
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('DT分类精度：%s' % score)
    # clf2 = AdaBoostClassifier(DecisionTreeClassifier(random_state=0), algorithm='SAMME')
    # # weights = weight_set(X_train)
    # clf2.fit(new_train, y_train)
    # score2 = clf2.score(new_test, y_test)
    # print('AdaBoost加权后的精度：%s' % score2)
    clf4 = SVC()
    clf4.fit(X_train, y_train)
    score4 = clf4.score(X_test, y_test)
    print('SVC分类精度：%s' % score4)
    # clf5 = AdaBoostClassifier(SVC(), algorithm='SAMME')
    # clf5.fit(new_train, y_train)
    # score5 = clf5.score(new_test, y_test)
    # print('AdaBoost加权后的精度：%s' % score5)

    # clf3 = AdaCostClassifier(DecisionTreeClassifier(random_state=0), algorithm='SAMME')
    # clf3.fit(new_train, y_train)
    # score3 = clf3.score(new_test, y_test)
    # print('AdaCost加权后的精度：%s' % score3)

    # #  获取特征importance
    # X_train, y_train, X_test, y_test = seven_train_thirty_test('./features/Harris_pca_feature.txt', './features/target', 10)
    # clf2 = AdaBoostClassifier(DecisionTreeClassifier(random_state=0), algorithm='SAMME')
    # weights = weight_set(X_train)
    # clf2.fit(X_train, y_train, weights)
    # get_importance(clf2, './features/Harris_pca_importance.txt')