# 导入所需模块 
from cv2 import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import util

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000
# 图像预处理
def img_preprocess(car_pic_file):
    """
    :param car_pic_file: 图像文件
    :return:已经处理好的图像文件 原图像文件
    """
    if type(car_pic_file) == type(""):
        img = util.img_read(car_pic_file)
    else:
        img = car_pic_file

    pic_hight, pic_width = img.shape[:2]
    if pic_width > MAX_WIDTH:
        resize_rate = MAX_WIDTH / pic_width
        img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
    # 缩小图片

    blur = 5
    img = cv2.GaussianBlur(img, (blur, blur), 0)
    oldimg = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("tmp/img_gray.jpg", img)
    # 转化成灰度图像
    util.plt_show_gray(img)

    Matrix = np.ones((20, 20), np.uint8)
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, Matrix)
    img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)
    # cv2.imwrite("tmp/img_opening.jpg", img_opening)
    # 创建20*20的元素为1的矩阵 开操作，并和img重合

    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_edge = cv2.Canny(img_thresh, 100, 200)
    # cv2.imwrite("tmp/img_edge.jpg", img_edge)
    # Otsu’s二值化 找到图像边缘
    util.plt_show_gray(img_edge)
    
    Matrix = np.ones((4, 19), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, Matrix)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
    util.plt_show_gray(img_edge2)

    return img_edge2, oldimg

# 图像定位与字符分割
def img_color_contours(img_contours, oldimg):
    """
    :param img_contours: 预处理好的图像
    :param oldimg: 原图像
    :return: 已经定位好的车牌
    """

    # if img_contours.any():
        #config.set_name(img_contours)
        # cv2.imwrite("tmp/img_contours.jpg", img_contours)

    pic_hight, pic_width = img_contours.shape[:2]

    card_contours = util.img_findContours(img_contours)
    # print('card_contours',card_contours)
    card_imgs = util.img_Transform(card_contours, oldimg, pic_width, pic_hight)
    # print('card_imgs',card_imgs)
    colors, car_imgs = util.img_color(card_imgs)
    roi = None
    card_color = None
    part_cards = ''
    # print('card_imgs',len(card_imgs))

    for i, color in enumerate(colors):
        if color in ("blue", "yello", "green"):
            card_img = card_imgs[i]
            # cv2.imwrite('./tmp/card_img'+str(i)+'.jpg', card_img)
            try:
                gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                # cv2.imwrite("tmp/card_gray_img.jpg", gray_img)

                # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
            except:
                raise ValueError('黄、绿车牌反向失败')
            if color == "green" or color == "yello":
                gray_img = cv2.bitwise_not(gray_img)
                # cv2.imwrite("tmp/card_gray_img2.jpg", gray_img)

            ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # cv2.imwrite("tmp/card_gray_img3.jpg", gray_img)

            x_histogram = np.sum(gray_img, axis=1)
            x_min = np.min(x_histogram)
            x_average = np.sum(x_histogram) / x_histogram.shape[0]
            x_threshold = (x_min + x_average) / 2

            wave_peaks = util.find_waves(x_threshold, x_histogram)
            if len(wave_peaks) == 0:
                # print("peak less 0:")
                continue
            # 认为水平方向，最大的波峰为车牌区域
            wave = max(wave_peaks, key=lambda x: x[1] - x[0])
            gray_img = gray_img[wave[0]:wave[1]]
            # cv2.imwrite("tmp/card_gray_img4.jpg", gray_img)

            # 查找垂直直方图波峰
            row_num, col_num = gray_img.shape[:2]
            # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
            gray_img = gray_img[1:row_num - 1]
            # cv2.imwrite("tmp/card_gray_img5.jpg", gray_img)

            y_histogram = np.sum(gray_img, axis=0)
            y_min = np.min(y_histogram)
            y_average = np.sum(y_histogram) / y_histogram.shape[0]
            y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半
            wave_peaks = util.find_waves(y_threshold, y_histogram)
            if len(wave_peaks) <= 6:
                # print("peak less 1:", len(wave_peaks))
                continue

            wave = max(wave_peaks, key=lambda x: x[1] - x[0])
            max_wave_dis = wave[1] - wave[0]
            # 判断是否是左侧车牌边缘
            if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                wave_peaks.pop(0)

            # 组合分离汉字
            cur_dis = 0
            for i, wave in enumerate(wave_peaks):
                if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                    break
                else:
                    cur_dis += wave[1] - wave[0]
            if i > 0:
                wave = (wave_peaks[0][0], wave_peaks[i][1])
                wave_peaks = wave_peaks[i + 1:]
                wave_peaks.insert(0, wave)
            point = wave_peaks[2]
            point_img = gray_img[:, point[0]:point[1]]
            if np.mean(point_img) < 255 / 5:
                wave_peaks.pop(2)

            if len(wave_peaks) <= 6:
                # print("peak less 2:", len(wave_peaks))
                continue

            part_cards = util.seperate_card(gray_img, wave_peaks)
            # print('part_cards',len(part_cards))
            # print(part_cards)
            roi = card_img
            card_color = color
            break

    return part_cards, roi, card_color  # 识别到的字符、定位的车牌图像、车牌颜色


