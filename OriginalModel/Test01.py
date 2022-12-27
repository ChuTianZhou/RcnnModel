import torch.nn as nn
from torch import nn
import os
import numpy as np
import torch
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.frcnn import FasterRCNN
from nets.frcnn_training import FasterRCNNTrainer, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch

if __name__ == '__main__':
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'
    with open(train_annotation_path) as f:
        # readlines() : 作为列表返回文件中的所有行，其中每一行都是列表对象中的一项：
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    train_dataset = FRCNNDataset(train_lines, [600, 600], train=True)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=2, num_workers=4, pin_memory=True,
                     drop_last=True, collate_fn=frcnn_dataset_collate)
    epoch_step = num_train // 2
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, boxes, labels = batch[0], batch[1], batch[2]
        print(labels)
    VOCdevkit_path = "DataSets"
    year = "2007"
    with open("DataSets/VOC2007/ImageSets/Main/test.txt") as file:
        test = file.read().strip().split()
    print(type(test))
    # num = 10
    # list = range(num)
    # print(type(list))

    # xmlfilepath = os.path.join("DataSets", 'VOC2007/Annotations')
    # # 返回所有xml文件的列表
    # temp_xml = os.listdir(xmlfilepath)
    # total_XML = []
    # for xml in temp_xml:
    #     # 如果以指定后缀结尾返回True，否则返回False
    #     if xml.endswith(".xml"):
    #         total_XML.append(xml)
    # name = total_XML[1][:-4] + '\n'
    # print(name)
    # VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]
    # for year, image_set in VOCdevkit_sets:
    #     image_ids = open(os.path.join("DataSets", 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)),
    #                      encoding='utf-8').read().strip().split()
    #     print(type(image_ids))

    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # x = np.array([0, 1, 2])
    # y = np.array([0, 1])
    #
    # X, Y = np.meshgrid(x, y)
    # print(X)
    # print(Y)
    #
    # plt.plot(X, Y,
    #          color='red',  # 全部点设置为红色
    #          marker='.',  # 点的形状为圆点
    #          linestyle='')  # 线型为空，也即点与点之间不用线连接
    # plt.grid(True)
    # plt.show()
    import torch
