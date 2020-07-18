
from PIL import Image
import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt
import os
# 之前的做法是直接将一张SAR影像进行（280*280）裁剪，经过上一次18/6月8号论文讨论班老师的意见，现将
# 算法改为overlapping方法对SAR影像进行剪切，同时边缘裁剪不足的地方由图像周围像素进行补充，取消原先
# 用白色填充的不足之处。


# 为能被280整除，所以需要对图像进行填充
def fill_image(image):
    height, width = image.shape
    print(width, height)

    new_image_length = width if width > height else height
    # new_image_width = 5766
    # new_image_height = 5952
    # print(new_image_length)

    # new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')
    # new_image = Image.new(image.mode, (new_image_width, new_image_height), color='white')

    # if width > height:
    #     new_image.paste(image, (0, int((new_image_length - height) / 2)))
    # else:
    #     new_image.paste(image, (int((new_image_length - width) / 2), 0))

    # 开始填充矩阵
    return image


# 对图像进行（280*280）分块处理
def cut_image(image):
    height, width = image.shape
    # item_width = int(width / 3)
    item_width = 280
    # 移动步长
    step_width = int(item_width/2)

    box_list = []
    count = 0
    max_width = 0
    max_height = 0
    for j in range(0, int((width-step_width)/step_width)):
        for i in range(0, int((height-step_width)/step_width)):
            count += 1
            box = image[i * step_width : i * step_width+item_width, j * step_width : j * step_width+item_width]
            # max_height = i * step_width+item_width
            # max_width = j * step_width+item_width
            io.imsave('F:\\论文\\download_SAR_data\\thirdExperimentCropImages\\'+str(i)+'行'+str(j)+'列'+'.tif', box)
            print('已保存第'+str(i)+'行'+str(j)+'列的图像')
            # box_list.append(box)
    print(count)
    # # 填充
    # pad_down = max_height - height
    # pad_right = max_width - width
    # new_image = cv2.copyMakeBorder(image, 0, pad_down, 0, pad_right, cv2.BORDER_REPLICATE)
    #
    # image_list = [new_image.crop(box) for box in box_list]


# 对分块后的各个图像进行保存
def save_images(image_list):
    index = 1
    for image in image_list:
        image.save('overlap_result/' + str(index) + '.tif')
        index += 1
        # print(image.size)


if __name__ == '__main__':
    file_path = '2.tif'
    # 打开图像，下面两个方法不适合大文件，所以才导致大文件无法读取
    # image = Image.open(file_path)
    # image = cv2.imread(file_path, 0)
    # 需要先设置内存限制，不然仍然会报错内存溢出
    Image.MAX_IMAGE_PIXELS = None
    image = io.imread(file_path)
    # 原始图像有两个波段，呈现的是tuple元祖的数据类型
    # 这里呢，先尝试采用第一个波段
    img_band0 = image[0]
    # img_show = io.imshow(img_band0)
    # plt.show()
    # 由于每个像素点值过大，导致无法正常显示，因此，这里对图像进行灰度处理
    # piexl_max = np.max(img_band0)
    # piexl_min = np.min(img_band0)
    # # 灰度变换
    # height = img_band0.shape[0]
    # width = img_band0.shape[1]
    # # 创建一幅图像
    # result = np.zeros((height, width))
    # for i in range(height):
    #     for j in range(width):
    #         piexl = img_band0[i][j]
    #         gray = int(piexl) / piexl_max
    #         result[i][j] = gray
    #         # print(result[i][j])
    # # 显示一下图像
    # img_show = io.imshow(result)
    # plt.show()
    # 将图像转为正方形，不够的地方补充为白色底色
    # image = fill_image(img_band0)
    # 分为图像
    image_list = cut_image(img_band0)
    # 保存图像
    # save_images(image_list)