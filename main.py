import cv2
import os
from extract_features import glcm_feature, harris_feature, lbp_feature, hu_feature
import fourier_descriptor
from HOG_feature import hog_feature

eddy_file = 'F:\\download_SAR_data\\experiment_data\\dataset\\eddy\\'
land_file = 'F:\\download_SAR_data\\experiment_data\\dataset\\land\\'
water_file = 'F:\\download_SAR_data\\experiment_data\\dataset\\sea_water\\'
if __name__ == '__main__':
    # 统计两个文件夹里面图片的个数
    COUNT_EDDY = len([name for name in os.listdir(eddy_file) if os.path.isfile(os.path.join(eddy_file, name))])
    COUNT_LAND = len([name for name in os.listdir(land_file) if os.path.isfile(os.path.join(land_file, name))])
    COUNT_WATER = len([name for name in os.listdir(water_file) if os.path.isfile((os.path.join(water_file, name)))])
    flag_eddy = flag_land = flag_water = 0
    target = []
    for i in range(COUNT_EDDY+COUNT_LAND+COUNT_WATER):
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
        #     contrast, correlation, energy, homogeneity = glcm_feature(image)
            # print(contrast, correlation, energy, homogeneity)
            # 用质心做的 傅里叶描述子
            # fourier_feature = fourier_descriptor.extract_feature(image)
            # print(fourier_feature.shape)
            # Harris特征
            # harris = harris_feature(image)
        #     LBP特征
        #     lbp = lbp_feature(image)
        #     print(i, len(lbp))
        #     Hu特征
            hu = hu_feature(image)
            print(i, len(hu))
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
            with open('./features/Hu_features.txt', 'a') as file:
                # file.write(str(contrast) + ' ' + str(correlation) + ' ' + str(energy) + ' ' + str(homogeneity) + ' ')
                # for j in range(67):
                #     file.write(str(fourier_feature[j])+' ')
                # for k in range(len(harris)):
                #     file.write(str(harris[k][0])+' ')
                # pca后的Harris
                # for k in range(len(harris[0])):
                #     file.write(str(harris[0][k]) + ' ')
                # for n in range(len(hog[0])):
                #     file.write(str(hog[0][n])+' ')
                # for l in range(len(lbp)):
                #     file.write(str(lbp[l])+' ')
                for h in range(len(hu)):
                    file.write(str(hu[h][0]) + ' ')
                file.write('\n')
            file.close()
        else:
            print('无法读取图片')
            # 对无法读取的图片应该不写入target中
            target.pop()
    print(target)