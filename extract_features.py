from skimage.feature import greycomatrix,greycoprops
import numpy as np
from numpy import *
import cv2
from skimage.exposure import equalize_hist
from skimage.feature import corner_harris, corner_peaks
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
import fourier_descriptor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

MIN_DESCRIPTOR = 18


def load_image(i):
    print('正在读取第' + str(i) + '个图片')
    img = cv2.imread('../KNN-GLCM/data/' + str(i) + '.tif', 0)
    if img is not None:
        img = cv2.resize(img, (280, 280), interpolation=cv2.INTER_CUBIC)
    else:
        print('无法读取第'+str(i)+'个图片')
    return img


# GLCM
def glcm_feature (img):
    glcms=greycomatrix(img,[1],[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)

    contrast = greycoprops(glcms,'contrast')
    sumCon = sumH = sumCor = sumE = 0

    for i in range(1):
        for j in range(4):
            sumCon = contrast[i][j] +sumCon
    averageCon = sumCon/4


    correlation = greycoprops(glcms,'correlation')
    for i in range(1):
        for j in range(4):
            sumCor = correlation[i][j] +sumCor
    averageCor = sumCor/4

    energy = greycoprops(glcms,'energy')
    for i in range(1):
        for j in range(4):
            sumE = energy[i][j] +sumE
    averageE = sumE/4

    homogeneity = greycoprops(glcms,'homogeneity')
    for i in range(1):
        for j in range(4):
            sumH = homogeneity[i][j] +sumH
    averageHomo = sumH/4
    return averageCon, averageCor, averageE, averageHomo


# harris特征
def harris_feature(img):
    mandrill = equalize_hist(rgb2gray(img))
    # corners = corner_peaks(corner_harris(mandrill), min_distance=1)
    # 使用corner_harris获取角点
    harris = corner_harris(mandrill)
    # 280*280 PCA降维 280*1
    pca = PCA(n_components=1)
    new_harris = pca.fit_transform(harris)
    return new_harris


# 写入文件
def feature_in_file(feature, file_path):
    with open(file_path, 'a') as file:
        for i in range(len(feature)):
            file.write(str(feature[i][0])+' ')
        file.write('\n')
    file.close()


# # 根据importance值来进行数据筛选
# def data_filter(feature):
#     new_data = []
#     # FD_importance
#     FD_importance = np.genfromtxt('./FD_importance.txt', dtype=str, delimiter=' ', usecols=range(280)).astype(float)
#     # Harris_importance
#     Harris_importance = np.genfromtxt('./Harris_importance.txt', dtype=str, delimiter=' ', usecols=range(280)).astype(float)
#     for i in range(len(FD_importance)):
#         if FD_importance[i]>0.001:
#             new_data
#     return


if __name__ == '__main__':
    for i in range(4938):
        image = load_image(i)
        if image is not None:
            fourier_feature = fourier_descriptor.extract_feature(image)
            # new_fourier = [[] for i in range(len(fourier_feature))]
            # for j in range(len(fourier_feature)):
            #     new_fourier[j].append(fourier_feature[j])
            # feature_in_file(fourier_feature, './features/FD_feature.txt')


            contrast, correlation, energy, homogeneity = glcm_feature(image)
            # # print('GLCM统计特征：对比度为%s，相关性为%s，能量为%s，同质度为%s' % (contrast, correlation, energy, homogeneity))
            # with open('./features/GLCM_feature.txt', 'a') as file:
            #     file.write(str(contrast) + ' ' + str(correlation)+' '+str(energy)+' '+str(homogeneity)+'\n')
            # file.close()
            #
            harris = harris_feature(image)
            # # print('第%s个Harris向量为：%s' % (i, harris))
            # feature_in_file(harris, './features/Harris_feature.txt')

            file = open('./features/FD_GLCM_Harris.txt', 'a')
            for j in range(len(fourier_feature)):
                file.write(str(fourier_feature[j])+' ')
            file.write(str(contrast)+' '+str(correlation)+' '+str(energy)+' '+str(homogeneity)+' ')
            for k in range(len(harris)):
                file.write(str(harris[k][0])+' ')
            file.write('\n')
            file.close()