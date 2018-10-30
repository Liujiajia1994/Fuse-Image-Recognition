import cv2
import numpy as np
import math

fd_len = 280
pi = 3.1415926


# # FD特征 傅立叶描述子Fourier descriptor
def fourier_descriptor_feature(img):
    contour = []
    binary, contour, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, contour)
    contour_array = contour[0][:, 0, :]
    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = np.fft.fft(contour_complex)
    return fourier_result
#
#
# def truncate_descriptor(descriptors, degree):
#     """
#     此函数截断未移位的傅立叶描述符数组并返回一个也没有移位
#
#     """
#     descriptors = np.fft.fftshift(descriptors)
#     center_index = len(descriptors) // 2
#     descriptors = descriptors[
#         center_index - degree // 2:center_index + degree // 2]
#     descriptors = np.fft.ifftshift(descriptors)
#     descriptors = np.absolute(descriptors)
#     return descriptors


def getCentroid(cnt):
    # (m, n) = pic.shape
    # r = 0
    # c = 0
    # count = 0
    # for i in range(0, m):
    #     for j in range(0, n):
    #         if pic[i][j] > 0:
    #             r += i
    #             c += j
    #             count += 1
    # return int(r / count), int(c / count)
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    print('the image coordinate parameter ', 'X:', cx)
    print('the image coordinate parameter ', 'Y:', cy)
    return cx, cy


def centroid_dist(boundarys, centroid):
    index = 0.0
    step = len(boundarys) / float(fd_len)
    dists = np.zeros(fd_len)
    for i in range(0, fd_len):
        j = int(index)
        dists[i] = eucld_metric(boundarys[j], centroid)
        index += step
    return dists


def eucld_metric(a, b):
    return math.sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]))


def FDFT(shape_signt):
    fds = np.zeros(fd_len)
    for n in range(0, fd_len):
        for i in range(0, fd_len):
            fds[n] += shape_signt[i] * math.exp(-2 * pi * i * n / fd_len)
        fds[n] /= fd_len
    return fds


def getContours(pic):
    img, contours, hrc = cv2.findContours(pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cts = np.array(contours[0])
    cts = cts.reshape(cts.shape[0], 2)
    return cts

# def getBoundary(pic):
#     edge = cv2.Canny(pic, 100, 250)
#     points = []
#     m, n = edge.shape
#     for i in range(0, m):
#         for j in range(0, n):
#             if edge[i][j] != 0:
#                 points.append((i, j))
#     return np.array(points)


def extract_feature(pic):
    # 获取轮廓
    contours = getContours(pic)
    # 获取质心
    ctr = getCentroid(pic)
    # the below func rise the accuracy about 3.5%
    # contours = getMaxContour(pic)
    # 计算轮廓到质心的距离
    shape_signt = centroid_dist(contours, ctr)
    # 傅立叶描述子
    fds = FDFT(shape_signt)
    # print(len(fds))
    return fds


def getMaxContour(pic):
    img, contours, hrc = cv2.findContours(pic, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    max_area = 0
    max_idx = 0
    area = 0
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max_area:
            max_idx = i
            max_area = area
    cts = np.array(contours[max_idx])
    cts = cts.reshape(cts.shape[0], 2)
    return cts


# if __name__ == '__main__':
#     for i in range(4938):
#         image = extract_features.load_image(i)
#         if image is not None:
#             fourier_feature = extract_feature(image)
#             if i % 2 == 0:
#                 extract_features.feature_in_file(fourier_feature, './eddy_FD_feature.txt')
#             else:
#                 extract_features.feature_in_file(fourier_feature, './other_FD_feature.txt')
#         else:
#             print('无法写入文件中！')