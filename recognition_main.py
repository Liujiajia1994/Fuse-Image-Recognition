import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

target_file = 'F:/download_SAR_data/experiment_data/dataset/target/target.txt'
data_file = './features/GLCM+FD+Harris_pca_features.txt'
if __name__ == '__main__':
    score_DT = score_SVC = score_KNN = 0
    data = np.genfromtxt(data_file, dtype=str, delimiter=' ', usecols=range(294)).astype(float)
    target = np.genfromtxt(target_file, dtype=str, delimiter=',', usecols=range(1)).astype(int)
    # 还是采用十折交叉验证来分类
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(data, target):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]
        # DT分类
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)
        print('DT分类精度：%s' % (clf.score(X_test, y_test)))
        score_DT += clf.score(X_test, y_test)
        # SVC分类
        clf4 = SVC()
        clf4.fit(X_train, y_train)
        print('SVC分类精度：%s' % (clf4.score(X_test, y_test)))
        score_SVC += clf4.score(X_test, y_test)
    #     KNN分类
        clf5 = KNeighborsClassifier(n_neighbors=2)
        clf5.fit(X_train, y_train)
        print('KNN分类精度：%s' % (clf5.score(X_test, y_test)))
        score_KNN += clf5.score(X_test, y_test)
    print('DT最后的分类精度：%s' % (score_DT/10))
    print('SVC最后的分类精度：%s' % (score_SVC/10))
    print('KNN最后的分类精度：%s' % (score_KNN/10))