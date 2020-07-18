import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import itemfreq
from extract_features import load_image,glcm_feature
from skimage.feature import hog
# 用于提取HOG特征


class Hog_descriptor():
    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img
        self.img = np.sqrt(img / np.max(img))
        self.img = img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size

    def extract(self):
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), int(self.bin_size)))
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)
        hog_vector = []
        # 判断hog_image是否为0矩阵，如果是0矩阵，就不做处理
        # 构造一个0矩阵，判断hog_image==0矩阵
        zero_array = np.zeros(hog_image.shape)
        if (hog_image == zero_array).all() == True:
            return None, None
        else:
            for i in range(cell_gradient_vector.shape[0] - 1):
                for j in range(cell_gradient_vector.shape[1] - 1):
                    block_vector = []
                    block_vector.extend(cell_gradient_vector[i][j])
                    block_vector.extend(cell_gradient_vector[i][j + 1])
                    block_vector.extend(cell_gradient_vector[i + 1][j])
                    block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                    mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                    magnitude = mag(block_vector)
                    if magnitude != 0:
                        normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                        block_vector = normalize(block_vector, magnitude)
                        hog_vector.append(block_vector)
            return hog_vector, hog_image

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) % self.bin_size, mod

    def render_gradient(self, image, cell_gradient):
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    if math.isnan(magnitude):
                        break
                    else:
                        x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                        y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                        x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                        y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                        cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                        angle += angle_gap
        return image


def hog_feature(image):
    hog = Hog_descriptor(image, cell_size=8, bin_size=9)
    vector, image_hog = hog.extract()
    if vector is None or image_hog is None:
        return None
    else:
        array_vector = np.array(vector)
        # # 对数组中nan值进行处理
        # for i in range(array_vector.shape[0]):
        #     for j in range(array_vector.shape[1]):
        #         if math.isnan(array_vector[i][j]):
        #             break
        # 要获取全部的特征还需要做归一化处理
        lbpTranpose = array_vector.transpose()
        pca = PCA(n_components=1)
        pca.fit(lbpTranpose)
        newData = pca.transform(lbpTranpose)
        hogTranpose = np.array(newData).transpose()
        return hogTranpose
    # orientations = 9
    # pixels_per_cell = [5, 5]
    # cells_per_block = [3, 3]
    # visualize = False
    # transform_sqrt = True
    # hog_feature = hog(image, orientations, pixels_per_cell, cells_per_block, visualise=visualize, transform_sqrt=transform_sqrt)
    # return hog_feature

# for i in range(1):
#     img = cv2.imread('./eddyData/train/eddy/'+str(i)+'.tif', cv2.IMREAD_GRAYSCALE)
#     imagere = cv2.resize(img, (280, 280), interpolation=cv2.INTER_CUBIC)
#     hog = Hog_descriptor(imagere, cell_size=8, bin_size=8)
#     vector, image = hog.extract()
#     print(np.array(vector).shape)
# # print(vector)
# # with open("HOG特征向量.txt", "a") as myfile:
# #     myfile.write(str(vector) + "\n")
# # histogram=itemfreq(image)
# # Calculate the histogram
# # x = itemfreq(image.ravel())
# # Normalize the histogram
# # hist = x[:, 1]/sum(x[:, 1])
# # 线图
# # plt.subplot(121)
# # plt.plot(hist)
# # 直方图
# # plt.subplot(121)
# # plt.hist(hist)
# # plt.subplot(122)
# #     plt.imshow(image, cmap='gray')
# #     plt.show()
#     lbpTranpose = np.array(vector).transpose()
#     pca = PCA(n_components=1)
#     pca.fit(lbpTranpose)
#     newData = pca.transform(lbpTranpose)
#     hogTranpose = np.array(newData).transpose()
#     print(hogTranpose.shape)
#     print("读写"+str(i)+"图片")
#     with open("HOG特征向量.txt", "a") as myfile:
#         for i in range(32):
#             myfile.write(str(hogTranpose[0][i]) + " ")
#         myfile.write("\n")
#     myfile.close()


# if __name__ =='__main__':
#     for i in range(4938):
#         image = load_image(i)
#         if image is not None:
#             hog = Hog_descriptor(image, cell_size=8, bin_size=9)
#             vector, image_hog = hog.extract()
#             array_vector = np.array(vector)
#             # 要获取全部的特征还需要做归一化处理
#             lbpTranpose = array_vector.transpose()
#             pca = PCA(n_components=1)
#             pca.fit(lbpTranpose)
#             newData = pca.transform(lbpTranpose)
#             hogTranpose = np.array(newData).transpose()
#             print(hogTranpose.shape)
#             # GLCM的4个特征
#             contrast, correlation, energy, homogeneity = glcm_feature(image)
#             # 写入文件
#             with open("./features/GLCM_HOG_pca_feature.txt", "a") as myfile:
#                 for i in range(36):
#                     myfile.write(str(hogTranpose[0][i]) + " ")
#                 myfile.write(str(contrast) + ' ' + str(correlation) + ' ' + str(energy) + ' ' + str(homogeneity) + ' ')
#                 myfile.write("\n")
#             myfile.close()
#             # print(array_vector.shape)
#             # normalised_blocks, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8),
#             #                                    block_norm='L2-Hys', visualise=True, feature_vector=True)
#             # print(normalised_blocks.shape)