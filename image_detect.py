import os
import cv2
from extract_features import glcm_feature, harris_feature
from fourier_descriptor import extract_feature
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold

def load_image_to_data():
    # 获取图像
    DIR = 'E:\\GitHub\\pyimgsaliency-master\\pyimgsaliency\\overlap_result'
    length = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    for i in range(length):
        # 输入图像
        print('正在读取第' + str(i + 1) + '个图片')
        image = cv2.imread(DIR + '\\' + str(i + 1) + '.tif', 0)
        if image is not None:
            # 提取特征
            contrast, correlation, energy, homogeneity = glcm_feature(image)
            fourier_feature = extract_feature(image)
            harris = harris_feature(image)
            # 写入文件
            file = open('./features/overlap_image_feature.txt', 'a')
            for j in range(len(fourier_feature)):
                file.write(str(fourier_feature[j]) + ' ')
            file.write(str(contrast) + ' ' + str(correlation) + ' ' + str(energy) + ' ' + str(homogeneity) + ' ')
            for k in range(len(harris)):
                file.write(str(harris[k][0]) + ' ')
            file.write('\n')
            file.close()
        else:
            print('第' + str(i + 1) + '张图片无法读取')


test_file = './features/overlap_image_feature.txt'
data_file = './features/FD_GLCM_Harris.txt'
target_file = './features/target'
if __name__ == '__main__':
    # load_image_to_data
    # 读取文件数据
    DTList = 0
    skf = StratifiedKFold(n_splits=10)
    X_data = np.genfromtxt(data_file, dtype=str, delimiter=' ', usecols=range(564)).astype(float)
    y_target = np.genfromtxt(target_file, dtype=str, delimiter=' ', usecols=range(1)).astype(int)
    # clf = DecisionTreeClassifier(random_state=0)
    # clf = SVC()
    # clf = neighbors.KNeighborsClassifier(n_neighbors=2)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    for train_index, test_index in skf.split(X_data, y_target):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_target[train_index], y_target[test_index]
        # 将这批数据进行分类
        clf.fit(X_train, y_train)
        DTList += clf.score(X_test, y_test)
        print('*****')
    print('训练精度：%s' % (DTList / 10))
    # 训练后用测试数据进行测试
    test_data = np.genfromtxt(test_file, dtype=str, delimiter=' ', usecols=range(564)).astype(float)
    print(clf.predict(test_data))