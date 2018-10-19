
from PIL import Image
import numpy as np

# 之前的做法是直接将一张SAR影像进行（280*280）裁剪，经过上一次18/6月8号论文讨论班老师的意见，现将
# 算法改为overlapping方法对SAR影像进行剪切，同时边缘裁剪不足的地方由图像周围像素进行补充，取消原先
# 用白色填充的不足之处。

# 为能被280整除，所以需要对图像进行填充
def fill_image(image):
    width, height = image.size
    print(width, height) # 5755*5940

    # new_image_length = width if width > height else height
    new_image_width = 5766
    new_image_height = 5952
    # print(new_image_length)

    # new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')
    new_image = Image.new(image.mode, (new_image_width, new_image_height), color='white')

    # if width > height:
    #     new_image.paste(image, (0, int((new_image_length - height) / 2)))
    # else:
    #     new_image.paste(image, (int((new_image_length - width) / 2), 0))
    return new_image

# 对图像进行（280*280）分块处理
def cut_image(image):
    width, height = image.size
    # item_width = int(width / 3)
    item_width = 280
    # 移动步长
    step_width = int(item_width/3)
    box_list = []
    count = 0
    for j in range(0, int(width/step_width)):
        for i in range(0, int(height/step_width)):
            count += 1
            box = (i * step_width, j * step_width, i * step_width+item_width, j * step_width+item_width)
            box_list.append(box)
    print(count)
    image_list = [image.crop(box) for box in box_list]
    return image_list

# 对分块后的各个图像进行保存
def save_images(image_list):
    index = 1
    for image in image_list:
        image.save('overlap_result/' + str(index) + '.tif')
        index += 1
        # print(image.size)

if __name__ == '__main__':
    file_path = "0.tif"
    # 打开图像
    image = Image.open(file_path)
    # 将图像转为正方形，不够的地方补充为白色底色
    # image = fill_image(image)
    # 分为图像
    image_list = cut_image(image)
    # 保存图像
    save_images(image_list)