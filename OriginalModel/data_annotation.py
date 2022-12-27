# first
import os
import random
# 用于读取xml文件
import xml.etree.ElementTree as ET

from utils.utils import get_classes

annotation_mode = 0
# 总体类别的个数
classes_path = 'model_data/data_classes.txt'
# --------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1 
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1 
#   仅在annotation_mode为0和1的时候有效
# --------------------------------------------------------------------------------------------------------------------------------#
trainval_percent = 0.9

train_percent = 0.9
# -------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
# -------------------------------------------------------#
Data_path = './DataSets'

Data_sets = ['train', 'val']
# 把我的要分类的类别的地址当作参数, 返回一个列表：分类名称
classes, _ = get_classes(classes_path)


def convert_annotation(image_id, list_file):
    # 这是一个xml文件的 相对路径： in_file
    in_file = open(os.path.join(Data_path, 'Annotations/%s.xml' % image_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()  # 获取根节点

    for obj in root.iter('object'):
        difficult = 0
        # difficult= 0 :代表好识别， 1 则不好识别
        if obj.find('difficult') is not None:
            # 获得 <difficult>的值
            difficult = obj.find('difficult').text
        # 获得 <name> 的值： 即分类的名称
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        # Element.find()只返回符合要求的第一个Element
        xmlbox = obj.find('bndbox')
        # b 是个集合，xmin,ymin,xmax,ymax
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    xmlfilepath = os.path.join(Data_path, 'Annotations')
    saveBasePath = os.path.join(Data_path, 'ImageSets/Main')
    # 返回所有xml文件的列表
    temp_xml = os.listdir(xmlfilepath)
    total_xml = []
    for xml in temp_xml:
        # 如果以指定后缀结尾返回True，否则返回False
        if xml.endswith(".xml"):
            total_xml.append(xml)

    num = len(total_xml)
    # list : (0, 10)  ----class 'range'
    list = range(num)
    # num * (训练集+验证集)与测试集的比例 : num * 0.9,再取整
    # tv 是训练集和验证集的个数
    tv = int(num * trainval_percent)
    # (训练集+验证集)中训练集与验证集的比例
    # tr 是训练集个数
    tr = int(tv * train_percent)
    # list是range(0,num)  从0到num，也就是从所有的xml文件中 随机取tv个数，返回的是个列表，其实是训练集和验证集
    trainval = random.sample(list, tv)
    # 从trainval这个列表中 随机取tr个数，当作训练集
    train = random.sample(trainval, tr)

    print("train and val size---->训练集和测试集的个数：", tv)
    print("train size---->训练集的格式：", tr)
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    # list : range(0, num)
    for i in list:
        # 切片 去掉.xml, 再加上 空行
        name = total_xml[i][:-4] + '\n'
        # 训练集+验证集
        #    训练集
        #     验证机
        #   测试集
        if i in trainval:
            # 是训练集或验证集
            ftrainval.write(name)
            if i in train:

                ftrain.write(name)
            else:
                fval.write(name)
        else:
            # 是测试集
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("ImageSets中 txt文件已生成")

    print("开始生成train.txt 和 val.txt : 地址+位置+标签 的txt文件")
    for image_set in Data_sets:
        image_ids = open(os.path.join(Data_path, 'ImageSets/Main/%s.txt' % image_set),
                         encoding='utf-8').read().strip().split()
        list_file = open('New_%s.txt' % image_set, 'w', encoding='utf-8')
        for image_id in image_ids:
            # os.path.abspath() : 获取该文件的绝对路径  E:\Test\faster-rcnn-pytorch-master\DataSets
            list_file.write('%s/JPEGImages/%s.jpg' % (os.path.abspath(Data_path), image_id))
            # 写入xmin,ymin,xmax,ymax, classes_id
            convert_annotation(image_id, list_file)
            list_file.write('\n')
        list_file.close()
    print("生成结束 train.txt 和 val.txt : 地址+位置+标签 的txt文件")
