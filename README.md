# chinese-license-plate-recognition
> 数字图像处理课程作业，车牌识别系统，非机器学习方法。

#### 运行步骤

下载数据集 ，模板文件。

按照如下位置放置文件

```
.
├── data_one
├── data_three
├── data_two
├── detectedPic
├── sourceFiles
├── templateFiles
├── util.py
├── imgLocationAndSplit.py
├── charRecognition.py
├── filterImg.py
├── log.txt
├── main.py
├── Dockerfile
└── requestments.txt

```

进入到程序目录后安装依赖`pip install -r requestments.txt`，运行`python3 main.py`。或者使用容器运行程序。

#### 文件说明

+ data_one，data_two，data_three 这是三个数据集文件，大概1400张车牌图像
+ templateFiles 存放字符模板数据集
+ detectedPic里存放的是三个数据集里可定位车牌的图像，大概370张
+ sourceFiles里存放的是车牌识别正确的图像以及字符识别率高的图像
+ imgLocationAndSplit.py用于车牌定位和字符分割
+ charRecognition.py 用于字符识别
+ filterImg.py 用于筛选图像
+ main.py 入口程序

