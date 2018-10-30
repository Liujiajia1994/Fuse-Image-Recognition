import cv2
import os
from extract_features import glcm_feature, harris_feature
import fourier_descriptor
from HOG_feature import hog_feature

eddy_file = 'F:\\download_SAR_data\\experiment_data\\dataset\\eddy\\'
land_file = 'F:\\download_SAR_data\\experiment_data\\dataset\\other\\'
if __name__ == '__main__':
    # 统计两个文件夹里面图片的个数
    COUNT_EDDY = len([name for name in os.listdir(eddy_file) if os.path.isfile(os.path.join(eddy_file, name))])
    COUNT_LAND = len([name for name in os.listdir(land_file) if os.path.isfile(os.path.join(land_file, name))])
    flag_eddy = flag_land = 0
    target = []
    for i in range(COUNT_EDDY+COUNT_LAND):
        if i % 2 == 0:
            print('正在读取eddy图片')
            image = cv2.imread(eddy_file + 'eddy-' + str(flag_eddy) + '.tif', 0)
            flag_eddy += 1
            target.append(1)
        else:
            print('正在读取other图片')
            image = cv2.imread(land_file + str(flag_land) + '.tif', 0)
            flag_land += 1
            target.append(0)
        if image is not None:
            # 统一尺寸
            image = cv2.resize(image, (280, 280), interpolation=cv2.INTER_CUBIC)
        #     提取灰度共生矩阵的4个统计特征
            contrast, correlation, energy, homogeneity = glcm_feature(image)
            # print(contrast, correlation, energy, homogeneity)
            # 用质心做的
            fourier_feature = fourier_descriptor.extract_feature(image)
            # print(fourier_feature)
            # Harris特征
            harris = harris_feature(image)
        #     HOG特征
        #     hog = hog_feature(image)
        #     if hog is None:
        #         print('无法获取该图片的HOG特征')
        #         # 不计入target中,z需要将其取出
        #         target.pop()
        #         break
        #     else:
            #     写入文件
            with open('./features/GLCM+FD+Harris_pca_features.txt', 'a') as file:
                file.write(str(contrast) + ' ' + str(correlation) + ' ' + str(energy) + ' ' + str(homogeneity) + ' ')
                for j in range(len(fourier_feature)):
                    file.write(str(fourier_feature[j])+' ')
                # for k in range(len(harris)):
                #     file.write(str(harris[k][0])+' ')
                for k in range(len(harris[0])):
                    file.write(str(harris[0][k]) + ' ')
                # for n in range(len(hog)):
                #     file.write(str(hog[n])+' ')
                file.write('\n')
            file.close()
        else:
            print('无法读取图片')
            # 对无法读取的图片应该不写入target中
            target.pop()
    print(target)