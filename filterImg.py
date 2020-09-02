# 导入所需模块 
from cv2 import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import imgLocationAndSplit
import charRecognition
import util
import json
import random

# 处理单个车牌图像
def car_lincese_recognition(pic_path):
    img_bgr = util.img_read(pic_path)
    processed_img, old_img = imgLocationAndSplit.img_preprocess(img_bgr)  # 预处理
    # util.plt_show_gray(processed_img)
    part_cards, roi, card_color =imgLocationAndSplit.img_color_contours(processed_img, old_img) # 定位与分割 
    # print(len(part_cards))
    results = charRecognition.template_matching(part_cards) # 检测
    # util.plt_show_color(roi)
    # for card in part_cards:
        # util.plt_show_gray(card)
    return "".join(results)

# 对图片list做预测
def recognition_list(img_list):
    num = 1
    total_char_cor_num = 0
    lincese_cor_num = 0
    total = len(img_list)
    random.shuffle(img_list)
    for img in img_list:
        ground_truth = img.split('/')[1][:7]
        recognition_res = car_lincese_recognition(img)[:7]        
        char_cor_num = 0
        for idx, char in enumerate(ground_truth):
            if char == recognition_res[idx]:
                total_char_cor_num += 1
                char_cor_num += 1
        if ground_truth == recognition_res:
            lincese_cor_num += 1
        print('ground_truth:{}, recognition_res:{}, cor_num:{}/7, {}/{}'.format(ground_truth, recognition_res, char_cor_num, num, total))        
        num += 1

    char_acc = total_char_cor_num / (total * 7)
    lincese_acc = lincese_cor_num / total 
    print('char_acc: {:0.2f}%'.format(char_acc * 100))
    print('lincese_acc: {:0.2f}%'.format(lincese_acc * 100))

# 识别一个数据集里可用的图片
def recognition_one_dataset(file_list_name):
    cor_img_list = []
    high_char_acc_img_list = []
    num = 0
    total_char_cor_num = 0
    lincese_cor_num = 0
    file_list = get_json_from_file(file_list_name)
    i = 0
    total = len(file_list)

    for filename in file_list:
        
        ground_truth = filename.split('/')[1][:7]
        recognition_res = car_lincese_recognition(filename)
        print('ground_truth:{} , recognition_res:{} , {}/{}'.format(ground_truth, recognition_res, i, total))
        
        num += 1
        char_cor_num = 0
        for idx, char in enumerate(ground_truth):
            if char == recognition_res[idx]:
                total_char_cor_num += 1
                char_cor_num += 1
        if char_cor_num >= 5:
            print('add high_char_acc_img_list')
            high_char_acc_img_list.append(filename)
        if ground_truth == recognition_res:
            print('add cor_img_list')
            lincese_cor_num += 1
            cor_img_list.append(filename)
        i += 1

    char_acc = total_char_cor_num / total * 7
    lincese_acc = lincese_cor_num / num 
    print('char_acc: %0.2f' % char_acc)
    print('lincese_acc: %0.2f' % lincese_acc)

    
    return cor_img_list, high_char_acc_img_list

# 检测单个数据集可用的图片
def detecte_pic(directory_name):
    imgs_list = []
    for filename in os.listdir(directory_name):
        file_path = directory_name + "/" + filename
        # print(file_path)
        img_bgr = util.img_read(file_path)
        processed_img, old_img = imgLocationAndSplit.img_preprocess(img_bgr)  # 预处理
        try:
            part_cards, roi, card_color = imgLocationAndSplit.img_color_contours(processed_img, old_img) # 定位与分割 
        except:
            continue
        if part_cards and roi.any() :
            imgs_list.append(directory_name+ '/' + filename)
    save_list_to_json('detectedPic/{}.json'.format(directory_name), imgs_list)
    
    return imgs_list

# 检测所有数据集可用的图片
def detecte_all_pic(directory_name_list):
    all_pic = []
    for directory_name in directory_name_list:
        res = detecte_pic(directory_name)
        all_pic.extend(res)
    return all_pic


# 存
def save_list_to_json(file_name, list):
    with open(file_name, 'w') as f:
        json.dump(list, f)

# 读
def get_json_from_file(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

# 对所有数据集可用的图片测试做预测
def recognition_all(dataset_config_file):
    all_cor_img_file = []
    high_char_acc_img_file = []
    for filename in dataset_config_file:
        cor_img_list, high_char_acc_img_list = recognition_one_dataset(filename)
        all_cor_img_file.extend(cor_img_list)
        high_char_acc_img_file.extend(high_char_acc_img_list)
    save_list_to_json('all_cor_img_file.json',all_cor_img_file)
    save_list_to_json('high_char_acc_img_file.json',high_char_acc_img_file)
    return all_cor_img_file, high_char_acc_img_file

if __name__ == '__main__':
    # window平台 目录名加 ./
    dataset_config_file = ['detectedPic/data_one.json', 'detectedPic/data_two.json', 'detectedPic/data_three.json']
    dataset_name = ['data_one', 'data_two', 'data_three'] 
    # 获得可用的图片
    # detecte_all_pic(dataset_name)
    # 对所有可用的图片测试 
    all_cor_img_file, high_char_acc_img_file =  recognition_all(dataset_config_file) # 返回预测正确的图片
    print(len(all_cor_img_file))  # 57
    print(len(high_char_acc_img_file))  # 185
