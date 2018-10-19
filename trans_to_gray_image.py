# 主要是用于将SAR图像转换成灰度图像，并进行保存
import os
import cv2
import matplotlib.pyplot as plt

if __name__ =='__main__':
    # 获取文件的个数
    DIR = 'C:\\Users\\lenovo\\Desktop\\SAR'
    length = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    for i in range(length):
        fig, ax = plt.subplots()
        print('正在读取第'+str(i)+'个图片')
        image = cv2.imread(DIR +'\\'+ str(i) + '.tif')
        if image is not None:
            image = cv2.resize(image, (280, 280), interpolation=cv2.INTER_CUBIC)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #     将SAR图像进行灰度显示
            plt.imshow(gray_image, cmap='gray')
            # 去掉坐标刻度
            plt.xticks([])
            plt.yticks([])
            # 去掉坐标轴
            plt.axis('off')
            # 去除图像周围的白边
            height, width = gray_image.shape
            # 如果dpi=300，那么图像大小=height*width
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
        #     将图像存放在文件夹里
        #     cv2.imwrite('C:\\Users\\lenovo\\Desktop\\grayEddyImage\\'+str(i)+'.tif', gray_image)
            # dpi是设置清晰度
            plt.savefig('C:\\Users\\lenovo\\Desktop\\grayEddyImage\\'+str(i)+'.tif', dpi=300)
            plt.show()
        else:
            print('无法保存第' + str(i) + '个图片')
