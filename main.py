import cv2
import os

from sklearn.decomposition import PCA

from extract_features import glcm_feature, harris_feature, lbp_feature, hu_feature
import fourier_descriptor
from HOG_feature import hog_feature
from pywavelet_feature import pywt_feature, gabor_feature

eddy_file = 'F:/论文/download_SAR_data/experiment_data/dataset/eddy/'
land_file = 'F:/论文/download_SAR_data/experiment_data/dataset/land/'
water_file = 'F:/论文/download_SAR_data/experiment_data/dataset/sea_water/'
if __name__ == '__main__':
    # 统计两个文件夹里面图片的个数
    COUNT_EDDY = len([name for name in os.listdir(eddy_file) if os.path.isfile(os.path.join(eddy_file, name))])
    # COUNT_EDDY = 1965
    COUNT_LAND = len([name for name in os.listdir(land_file) if os.path.isfile(os.path.join(land_file, name))])
    # COUNT_LAND = 1965
    COUNT_WATER = len([name for name in os.listdir(water_file) if os.path.isfile((os.path.join(water_file, name)))])
    # COUNT_WATER = 1965
    flag_eddy = flag_land = flag_water = 0
    target = []
    glcm_array = [[0 for i in range(4)] for j in range(4)]
    for i in range(5895):
        if i % 3 == 0:
            print('正在读取eddy-' + str(flag_eddy) + '.tif图片')
            image = cv2.imread(eddy_file + 'eddy-' + str(flag_eddy) + '.tif', 0)
            flag_eddy += 1
            target.append(0)
        elif i % 3 == 1:
            print('正在读取land'+str(flag_land) + '.tif图片')
            image = cv2.imread(land_file + 'land-' + str(flag_land) + '.tif', 0)
            flag_land += 1
            target.append(1)
        else:
            print('正在读取water'+str(flag_water) + '.tif图片')
            image = cv2.imread(water_file + 'water-' + str(flag_water) + '.tif', 0)
            flag_water += 1
            target.append(2)
        if image is not None:
            # 统一尺寸
            image = cv2.resize(image, (67, 67), interpolation=cv2.INTER_CUBIC)
        #     提取灰度共生矩阵的4个统计特征
            contrast, correlation, energy, homogeneity = glcm_feature(image)

            # 取消mean降维，采用PCA降维，尝试结果
            for p in range(4):
                glcm_array[p][0] = contrast[0][p]
                glcm_array[p][1] = correlation[0][p]
                glcm_array[p][2] = energy[0][p]
                glcm_array[p][3] = homogeneity[0][p]
            #
            # pca = PCA(n_components=1)
            # new_glcm = pca.fit_transform(glcm_array)


            # print(contrast, correlation, energy, homogeneity)
            # 用质心做的 傅里叶描述子
            fourier_feature = fourier_descriptor.extract_feature(image)
            # print(fourier_feature.shape)
            # Harris特征
            # harris = harris_feature(image)
        #     LBP特征
        #     lbp = lbp_feature(image)
        #     print(i, len(lbp))
        #     小波变换 纹理特征
        #     pywt = pywt_feature(image)
            # print(len(pywt))
            #     for w in range(len(pywt[p])):
            #         print(len(pywt[p][w]))
        #     Gabor小波 纹理
        #     gabor = gabor_feature(image)
        #     Hu特征
        #     hu = hu_feature(image)
        #     print(i, len(hu))
        #     HOG特征
        #     hog = hog_feature(image)
        #     if hog is None:
        #         print('无法获取该图片的HOG特征')
        #         # 不计入target中,z需要将其取出
        #         target.pop()
        #         break
        #     else:
        #         print('写入HOG文件')
        #         print(hog.shape)
        #     写入文件
            with open('./features/GLCM_FD_features.txt', 'a') as file:
                # file.write(str(contrast) + ' ' + str(correlation) + ' ' + str(energy) + ' ' + str(homogeneity) + ' ')
                for p in range(4):
                    for q in range(4):
                        file.write(str(glcm_array[p][q])+' ')
                for j in range(67):
                    file.write(str(fourier_feature[j])+' ')
                # for k in range(len(harris)):
                #     file.write(str(harris[k][0])+' ')
                # tradition method
                # for k in range(harris.shape[0]):
                #     for n in range(harris.shape[1]):
                #         file.write(str(harris[k][n])+' ')
                # pca后的Harris
                # for k in range(len(harris[0])):
                #     file.write(str(harris[0][k]) + ' ')
                # for n in range(len(hog[0])):
                #     file.write(str(hog[0][n])+' ')
                # for l in range(len(lbp)):
                #     file.write(str(lbp[l])+' ')
                # for h in range(len(hu)):
                #     file.write(str(hu[h][0]) + ' ')
                # for p in range(len(pywt)):
                #     for w in range(len(pywt[p])):
                #         file.write(str(pywt[p][w]))
                file.write('\n')
            file.close()
        else:
            print('无法读取图片')
            # 对无法读取的图片应该不写入target中
            target.pop()
    print(target)