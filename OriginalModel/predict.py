import time
import 123
import cv2
import numpy as np
from PIL import Image
from frcnn import FRCNN

if __name__ == "__main__":
    frcnn = FRCNN()
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等
    #   'dir_predict'表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹
    mode = "predict"
    #   crop指定了是否在单张图片预测后对目标进行截取
    #   crop仅在mode='predict'时有效
    crop = False
    test_interval = 100
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            # 没有出现异常执行
            else:
                # crop ： 是否进行裁剪 False
                r_image = frcnn.detect_image(image, crop=crop)
                r_image.show()
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = frcnn.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
