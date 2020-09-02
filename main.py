import filterImg
import os
import platform


linux_directory_name = 'sourceFiles'
windows_directory_name = './sourceFiles'


if platform.system().lower() == 'windows':
    PATH = windows_directory_name
elif platform.system().lower() == 'linux' or platform.system().lower() == 'darwin':
    PATH = linux_directory_name

# note: window下运行程序 请更改main.py和charRecognition.py内的PATH为window_directory_name

# linux: directory_name = 'sourceFiles' , window: directory_name = './sourceFiles'
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



imgs_list = get_imgs_list(directory_name=PATH,imgs_num=10)
# print(len(imgs_list))
filterImg.recognition_list(imgs_list)
lis = filterImg.get_json_from_file('./sourceFiles/all_cor_img_file.json')
print(len(lis))