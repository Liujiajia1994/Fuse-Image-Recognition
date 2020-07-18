# alt+enter可以导包
import cv2
import numpy as np
from scipy.misc import imresize
from PIL import Image
import os
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import random

data_file = 'F:\\download_SAR_data\\experiment_data\\no_processed_big_water_data\\'
save_file = 'F:\\download_SAR_data\\experiment_data\\dataset\\sea_water\\'


# 裁剪图片的四个71*71，以及中间的一块
def defined_crop(i, img):
    h, w = img.shape
    print('裁剪第'+str(i)+'张图片')
    crop_img1 = img[1:int(3*h/4), 1:int(3*w/4)]
    crop_img2 = img[int(h/4):h, 1:int(3*w/4)]
    crop_img3 = img[1:int(3*h/4), int(w/4):w]
    crop_img4 = img[int(h/4):h, int(w/4):w]
    # crop_img5 = img[int(h/4):int(3*h/4), int(w/4):int(3*w/4)]
    # plt.subplot(161)
    # plt.imshow(img, cmap='gray')
    # plt.subplot(162)
    # plt.imshow(crop_img1, cmap='gray')
    # plt.subplot(163)
    # plt.imshow(crop_img2, cmap='gray')
    # plt.subplot(164)
    # plt.imshow(crop_img3, cmap='gray')
    # plt.subplot(165)
    # plt.imshow(crop_img4, cmap='gray')
    # plt.subplot(166)
    # plt.imshow(crop_img5, cmap='gray')
    # plt.show()
    print('保存第'+str(i)+'张图片')
    cv2.imwrite(save_file + str(i) + 'crop-1.tif', crop_img1)
    cv2.imwrite(save_file + str(i) + 'crop-2.tif', crop_img2)
    cv2.imwrite(save_file + str(i) + 'crop-3.tif', crop_img3)
    cv2.imwrite(save_file + str(i) + 'crop-4.tif', crop_img4)
    # cv2.imwrite(save_file + str(i) + 'crop-5.tif', crop_img5)


def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size


# 随机裁剪
def random_crop(image, crop_size):
    crop_size = check_size(crop_size)
    h, w = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    # # 随机生成不重复的整数
    # top = random.sample(range(1, 9), 1)[0]*10
    # left = random.sample(range(1, 9), 1)[0]*10
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right]
    return image


def center_crop(image, crop_size):
    crop_size = check_size(crop_size)
    h, w = image.shape
    top = (h - crop_size[0]) // 2
    left = (w - crop_size[1]) // 2
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right]
    return image


def save_image(image, imagefile):
    image = np.asarray(image, dtype=np.uint8)
    image = Image.fromarray(image)
    image.save(imagefile)


# 文件重命名
def file_rename():
    count = 0
    path = save_file
    filelist = os.listdir(path)#该文件夹下所有的文件（包括文件夹）
    for files in filelist:#遍历所有文件
        Olddir = os.path.join(path,files)#原来的文件路径
        if os.path.isdir(Olddir):#如果是文件夹则跳过
            continue
        filename = os.path.splitext(files)[0]#文件名
        filetype = os.path.splitext(files)[1]#文件扩展名
        Newdir = os.path.join(path, 'water-'+str(count)+filetype)#新的文件路径
        os.rename(Olddir,Newdir)#重命名
        count += 1


# 尺度变换
def scale_augmentation(image, i, crop_size):
    # scale_size = np.random.randint(*scale_range)
    scale_size = int((0.2*i+1)*crop_size)
    image = imresize(image, (scale_size, scale_size))
    # image = random_crop(image, crop_size)
    image = center_crop(image, crop_size)
    return image


def resize(image, size):
    size = check_size(size)
    image = imresize(image, size)
    return image


# 旋转
def random_rotation(i,image):
    h, w = image.shape
    angle = i*30
    image = rotate(image, angle)
    image = resize(image, (h, w))
    return image


# 生成5张图片
def generate_image(i, image, arg):
    count = 1
    # crop_size = image.shape[0] if image.shape[0] < image.shape[1] else image.shape[1]
    while 1:
        # if( arg == 'rotation'):
        #     images = random_rotation(count, image)
        # else:
        #     images = scale_augmentation(image, count, crop_size)
        images = random_crop(image, 280)
        # # 显示图片
        # plt.subplot(121)
        # plt.imshow(image, cmap='gray')
        # plt.subplot(122)
        # plt.imshow(images, cmap='gray')
        # plt.show()
        print('正在保存第' + str(i) + '-' + str(count) + '张图片')
        cv2.imwrite(save_file + str(i) + arg + '-' + str(count) + '.tif', images)
        count += 1
        if count == 7:
            break


def read_shape():
    for i in range(1965):
        print('正在读取第'+str(i)+'张图片')
        image = cv2.imread(save_file + 'eddy-'+str(i) + '.tif', 0)
        if image is not None:
            print('第' + str(i) + '张图片为', image.shape)
        else:
            print('无法读取第'+str(i)+'张图片')


if __name__ == '__main__':
    file_rename()
    # read_shape()
    # for i in range(15):
    #     print('正在读取第'+str(i+1)+'张图片')
    #     image = cv2.imread(data_file + str(i+1) + '.tif', 0)
    #     generate_image(i, image, 'water')
    # #     if image is not None:
    # #         # defined_crop(i, image)
    # #         # generate_image(i, image, 'rotation')
    # #         # generate_image(i, image, 'scale')
    # #     else:
    # #         print('无法读取第'+str(i)+'张图片')
    # #     i += 2


