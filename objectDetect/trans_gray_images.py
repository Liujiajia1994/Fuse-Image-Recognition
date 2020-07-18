import numpy as np
import os
import cv2
from skimage import io
import matplotlib.pyplot as plt


# 对裁剪后的图片进行灰度转化
def trans_gray():
    file_path = 'F:/download_SAR_data/experiment_data/eddy/'
    # count = len([name for name in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, name))])
    for name in os.listdir(file_path):
        # count = len(name)
        print('正在读取' + name + '图片')
        imagePath = file_path+name
        image = cv2.imread(imagePath)
        piexl_max = np.max(image)
        piexl_min = np.min(image)
        # 灰度变换
        height = image.shape[0]
        width = image.shape[1]
        # 创建一幅图像
        result = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                piexl = image[i][j]
                # gray = int(piexl) / piexl_max
                result[i][j] = piexl / piexl_max * 255
                # print(result[i][j])
        # 显示一下图像
        io.imshow(result)
        plt.show()
        # 保存图片
        io.imsave('F:\\download_SAR_data\\thirdExp\grayImages\\' + name, result)
        print('已保存第' + name + '列的图像')


if __name__ == '__main__':
    trans_gray()