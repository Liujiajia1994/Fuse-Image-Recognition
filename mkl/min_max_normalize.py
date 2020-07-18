# 三种特征分别归一化，然后串联后再归一化，再进行多核学习
import numpy as np
from sklearn import preprocessing

GLCM_file = 'F:/GitHub/Fuse-Image-Recognition/features/GLCM_features.txt'
FD_file = 'F:/GitHub/Fuse-Image-Recognition/features/FD_features.txt'
Harris_file = 'F:/GitHub/Fuse-Image-Recognition/features/Harris_features.txt'
data_file = 'F:/GitHub/Fuse-Image-Recognition/features/GLCM+FD+Harris_features.txt'


def normalize(file):
    # 读数据文件
    data = np.genfromtxt(file, dtype=str, delimiter=' ').astype(float)
    #     进行最大最小归一化处理
    feature = preprocessing.MinMaxScaler().fit_transform(data)
    with open('F:/GitHub/Fuse-Image-Recognition/features/GLCM+FD+Harris_normalized_features.txt', 'a+') as file:
        for i in range(feature.shape[0]):
            for j in range(feature.shape[1]):
                file.write(str(feature[i][j])+' ')
            file.write('\n')
        file.close()


if __name__ == '__main__':
    #  这个是每个文件进行最大最小归一化处理
    # normalize(GLCM_file)
    # normalize(FD_file)
    # normalize(Harris_file)
    # 接下来做的是对三个文件合在一起进行归一化处理
    # GLCM_normal_file = 'F:/GitHub/Fuse-Image-Recognition/features/GLCM_normalize_features.txt'
    # FD_normal_file = 'F:/GitHub/Fuse-Image-Recognition/features/FD_normalize_features.txt'
    # Harris_normal_file = 'F:/GitHub/Fuse-Image-Recognition/features/Harris_normalize_features.txt'
    # GLCM_normal_data = np.genfromtxt(GLCM_normal_file, dtype=str, delimiter=' ').astype(float)
    # FD_normal_data = np.genfromtxt(FD_normal_file, dtype=str, delimiter=' ').astype(float)
    # Harris_normal_data = np.genfromtxt(Harris_normal_file, dtype=str, delimiter=' ').astype(float)
    # with open('F:/GitHub/Fuse-Image-Recognition/features/GLCM+FD+Harris_normalize_features.txt', 'a+') as file:
    #     for i in range(GLCM_normal_data.shape[0]):
    #         for j in range(GLCM_normal_data.shape[1]):
    #             file.write(str(GLCM_normal_data[i][j])+' ')
    #         for k in range(FD_normal_data.shape[1]):
    #             file.write(str(FD_normal_data[i][k])+' ')
    #         for l in range(Harris_normal_data.shape[1]):
    #             file.write(str(Harris_normal_data[i][l])+' ')
    #         file.write('\n')
    #     file.close()
    file = 'F:/GitHub/Fuse-Image-Recognition/features/GLCM+FD+Harris_normalize_features.txt'
    normalize(file)



