import cv2
import numpy as np
import pywt
from skimage import filters


def pywt_feature(image):
    # 小波变换 视觉信号 也是纹理特征的一种
    coeffs = pywt.wavedec(image, 'db1', level=2)
    # 返回结果为level+1个数字，第一个数组为逼近系数数组，后面的依次是细节系数数组
    # features = np.array(coeffs)
    return coeffs


def gabor_feature(image):
    filt_real, filt_imag = filters.gabor_filter(image, frequency=0.6)
    return filt_real
#
# if __name__ == '__main__':
#     eddy_file = 'F:\\download_SAR_data\\experiment_data\\dataset\\eddy\\'
#     img = cv2.imread(eddy_file + 'eddy-' + str(1) + '.tif', 0)


