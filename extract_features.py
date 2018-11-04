from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
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
from datetime import datetime

MIN_DESCRIPTOR = 18
eddy_file = 'F:\\download_SAR_data\\experiment_data\\dataset\\eddy\\'
land_file = 'F:\\download_SAR_data\\experiment_data\\dataset\\land\\'
# settings for LBP
radius = 1
n_points = 8*radius


def load_image(i):
    print('正在读取第' + str(i) + '个图片')
    img = cv2.imread(eddy_file + 'eddy-' + str(i) + '.tif', 0)
    if img is not None:
        img = cv2.resize(img, (67, 67), interpolation=cv2.INTER_CUBIC)
    else:
        print('无法读取第'+str(i)+'个图片')
    return img


# GLCM
def glcm_feature(img):
    glcms = greycomatrix(img,[1],[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)

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

    # # PCA
    # pca = PCA(n_components=10)
    # pca.fit(harris)
    # pcaharris = pca.transform(harris)
    # # print(pcaharris.shape)
    # pcaharris = np.array(pcaharris).transpose()
    # # print(pcaharris.shape)
    #
    # pca = PCA(n_components=1)
    # pca.fit(pcaharris)
    # pcaharris = pca.transform(pcaharris)
    # pcaharris = np.array(pcaharris).transpose()
    # return pcaharris


# LBP特征
def lbp_feature(image):
    lbp = local_binary_pattern(image, n_points, radius,method='uniform')
# 1,8,ror = 256; 1,8,uniform = 10;1,8,nri_uniform = 59;
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
    return hist


# Hu不变矩特征
def hu_feature(image):
    moments = cv2.moments(image)
    humoments = cv2.HuMoments(moments)
    humoments = np.log(np.abs(humoments))  # 同样建议取对数
    # print(humoments)
    return humoments


# 写入文件
def feature_in_file(feature, file_path):
    with open(file_path, 'a') as file:
        for i in range(len(feature[0])):
            file.write(str(feature[0][i])+' ')
            # print(str(feature[i])+' ')
        file.write('\n')
    file.close()


# if __name__ == '__main__':
    # t1 = datetime.now()
    # for i in range(1964):
    #     image = load_image(i)
    #     if image is not None:
    #         # Hu不变矩
    #         hu = hu_feature(image)  # 7*1
    #         print(len(hu))
    #         for h in range(len(hu)):
    #             print(hu[h][0])
            # fourier_feature = fourier_descriptor.extract_feature(image)  # 280
#             # new_fourier = [[] for i in range(len(fourier_feature))]
#             # for j in range(len(fourier_feature)):
#             #     new_fourier[j].append(fourier_feature[j])
#             # feature_in_file(fourier_feature, './features/FD_feature.txt')
#
#             contrast, correlation, energy, homogeneity = glcm_feature(image)  # 4
#             # # print('GLCM统计特征：对比度为%s，相关性为%s，能量为%s，同质度为%s' % (contrast, correlation, energy, homogeneity))
#             # with open('./features/GLCM_feature.txt', 'a') as file:
#             #     file.write(str(contrast) + ' ' + str(correlation)+' '+str(energy)+' '+str(homogeneity)+'\n')
#             # file.close()
#             #
#             harris = harris_feature(image)
#             # harris = pca_harris(image)  # 10
#             # # print('第%s个Harris向量为：%s' % (i, harris))
#             # feature_in_file(harris, './features/Harris_pca_feature.txt')
#
#             file = open('./features/FD_GLCM_Harris_pca.txt', 'a')
#             for j in range(len(fourier_feature)):
#                 file.write(str(fourier_feature[j])+' ')
#             file.write(str(contrast)+' '+str(correlation)+' '+str(energy)+' '+str(homogeneity)+' ')
#             for k in range(len(harris[0])):
#                 file.write(str(harris[0][k])+' ')
#             file.write('\n')
#             file.close()
#             print(datetime.now() - t1)