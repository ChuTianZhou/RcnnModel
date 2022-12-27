import os
import cv2
import pydicom
import datetime
from tqdm import tqdm


def save_label_img(filename, ID):
    try:
        dcm = pydicom.read_file(filename)
    except AttributeError:
        return
    info = {
        'PatientName': str(dcm.PatientName.decode('gbk')),
        'PatientSex': dcm.PatientSex,
        'PatientBirthDate': dcm.PatientBirthDate,
        'img_gray': dcm.pixel_array,
        'StudyDate': dcm.StudyDate
    }
    birth_day = datetime.datetime.strptime(info['PatientBirthDate'], '%Y%m%d')
    study_day = datetime.datetime.strptime(info['StudyDate'], '%Y%m%d')
    age = study_day.__sub__(birth_day).days // 365
    pic_name = '../Proce_Pic2/' + str(ID).zfill(5) + info['PatientSex'] + str(age) + '.jpg'
    cv2.imwrite(pic_name, info['img_gray'])
    print('完成对' + filename + '的解析和保存')


def walk_transform():
    ID = 1894
    for j in range(24):
        tmp_loc = ''
        if j + 1 < 10:
            tmp_loc = r'D:\CT\disk2\PAT0000' + str(j + 1) + r'\STD00001'
        else:
            tmp_loc = r'D:\CT\disk2\PAT000' + str(j + 1) + r'\STD00001'
        num = len(os.listdir(tmp_loc)) - 1
        for i in range(num):
            if i + 1 < 10:
                root = tmp_loc + '\SER0000' + str(i + 1)
            else:
                root = tmp_loc + '\SER000' + str(i + 1)
            files = os.listdir(root)
            for file in files:
                if file[-3:] != 'dir':
                    try:
                        save_label_img(os.path.join(root, file), ID)
                    except AttributeError:
                        pass
                    ID = ID + 1


def is_have_nose(img):
    height, width = img.shape[0:2]
    max_height = 0
    min_height = 0
    for row in range(height):
        for col in range(width):
            if max_height == 0 and img[row, col] != 0:
                max_height = row
            if min_height == 0 and img[height - row - 1, col] != 0:
                min_height = height - row
    if abs(max_height - min_height) > 55 and max_height > 20:
        return True
    else:
        return False


def cut_pic():
    root = r'../Proce_Pic'
    pic_list = os.listdir(root)
    for index, filename in enumerate(pic_list):
        if index < 100:
            print("正在处理第{}张图片".format(index))
            pic = cv2.cvtColor(cv2.imread(
                os.path.join(root, os.path.join(root, filename))), cv2.COLOR_BGR2GRAY)
            ymin = 171
            xmin = 120
            w = 130
            h = 130
            imgCrop = pic[xmin:xmin + h, ymin:ymin + w].copy()  # 切片获得裁剪后保留的图像区域
            _, dst = cv2.threshold(imgCrop, 170, 255, cv2.THRESH_BINARY)  # 分割图像二值化处理
            if is_have_nose(dst):
                cv2.imwrite(r'../Proc_Pic_CAJ/' + filename, imgCrop)


def look_threshold():
    img = cv2.cvtColor(
        cv2.imread(r'C:\Users\zct\PycharmProjects\torch\faster-rcnn-pytorch-master\Proce_Pic\00948F40.jpg'),
        cv2.COLOR_BGR2GRAY)
    ymin = 181
    xmin = 130
    w = 120
    h = 120
    imgCrop = img[xmin:xmin + h, ymin:ymin + w].copy()
    _, dst = cv2.threshold(imgCrop, 180, 255, cv2.THRESH_BINARY)
    cv2.imshow('test', dst)
    cv2.waitKey(0)
    is_have_nose(dst)


cut_pic()