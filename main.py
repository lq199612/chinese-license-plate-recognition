import filterImg
import os
import platform


linux_directory_name = 'sourceFiles'
windows_directory_name = './sourceFiles'

# 设置不同系统下的directory_name
if platform.system().lower() == 'windows':
    PATH = windows_directory_name
elif platform.system().lower() == 'linux' or platform.system().lower() == 'darwin':
    PATH = linux_directory_name



def get_imgs_list(directory_name = 'sourceFiles', default_file_name = 'all_cor_img_file.json',imgs_num=100):
    imgs_list = []
    files = os.listdir(directory_name)
    default_imgs_list = filterImg.get_json_from_file(directory_name + '/' + default_file_name)
    imgs_list.extend(default_imgs_list)
    imgs_num -= len(default_imgs_list)
    if imgs_num <= 0:
        return default_imgs_list[:imgs_num]
    files.remove(default_file_name)
    for file in files:
        file_imgs_list = filterImg.get_json_from_file(directory_name + '/' + file)
        if len(file_imgs_list) >= imgs_num:
            imgs_list.extend(file_imgs_list[:imgs_num])
            break
        else:
            imgs_list.extend(file_imgs_list)
            imgs_num -= len(file_imgs_list)
    return imgs_list


# 获取图像列表
imgs_list = get_imgs_list(directory_name=PATH,imgs_num=10)
# 做检测
filterImg.recognition_list(imgs_list)