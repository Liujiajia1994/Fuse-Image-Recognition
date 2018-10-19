# alt+enter可以导包
import cv2
import numpy as np
from scipy.misc import imresize
from PIL import Image
import os
import matplotlib.pyplot as plt


def defined_crop():
    for i in range(131):
        print("正在读取" + str(i) + "个图片")
        img = cv2.imread('./eddyData/train/ice/' + str(i) + '.tif')
        img = cv2.resize(img, (280, 280), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crop_img1 = img[1:140, 1:140]
        crop_img2 = img[140:280, 1:140]
        crop_img3 = img[1:140, 140:280]
        crop_img4 = img[140:280, 140:280]
        crop_img5 = img[70:210, 70:210]
        cv2.imwrite('./eddyData/train/iceCrop/' + str(i) + '-1.tif', crop_img1)
        cv2.imwrite('./eddyData/train/iceCrop/' + str(i) + '-2.tif', crop_img2)
        cv2.imwrite('./eddyData/train/iceCrop/' + str(i) + '-3.tif', crop_img3)
        cv2.imwrite('./eddyData/train/iceCrop/' + str(i) + '-4.tif', crop_img4)
        cv2.imwrite('./eddyData/train/iceCrop/' + str(i) + '-5.tif', crop_img5)


def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size


def random_crop(image, crop_size):
    crop_size = check_size(crop_size)
    h, w = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right]
    return image


def save_image(image, imagefile):
    image = np.asarray(image, dtype=np.uint8)
    image = Image.fromarray(image)
    image.save(imagefile)


def file_rename():
    count = 0
    path = "F:\\download_SAR_data\\experiment_data\\sea_water\\"
    filelist = os.listdir(path)#该文件夹下所有的文件（包括文件夹）
    for files in filelist:#遍历所有文件
        Olddir = os.path.join(path,files)#原来的文件路径
        if os.path.isdir(Olddir):#如果是文件夹则跳过
            continue
        filename = os.path.splitext(files)[0]#文件名
        filetype = os.path.splitext(files)[1]#文件扩展名
        Newdir = os.path.join(path,str(count)+filetype)#新的文件路径
        os.rename(Olddir,Newdir)#重命名
        count += 1


if __name__ == '__main__':
    file_rename()
    # data_file = 'F:\\download_SAR_data\\experiment_data\\no_processed_big_land_data\\'
    # for i in range(8):
    #     print('正在读取第'+str(i+1)+'张图片')
    #     image = cv2.imread(data_file+str(i+1)+'.tif')
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     # image_size = resize(image,280)
    #     print('正在随机剪切第' + str(i + 1) + '张图片')
    #     count = 1
    #     while 1:
    #         image_crop = random_crop(image, 280)
    #         # 显示图片
    #         # plt.subplot(151)
    #         # plt.imshow(image_crop, cmap='gray')
    #         # plt.show()
    #         print('正在保存第' + str(i+1)+'-'+str(count) + '张图片')
    #         # save_image(image_crop, 'F:\\download_SAR_data\\experiment_data\\land\\test_data\\'+str(i+1)+'.tif')
    #         cv2.imwrite('F:\\download_SAR_data\\experiment_data\\land\\'+str(i+1)+'-'+str(count)+'.tif', image_crop)
    #         count += 1
    #         if count == 251:
    #             break
