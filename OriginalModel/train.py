import numpy as np
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

if __name__ == "__main__":
    Cuda = False
    classes_path = 'model_data/data_classes.txt'
    model_path = 'model_data/voc_weights_resnet.pth'  # 给出预训练权重
    #   输入的shape大小
    input_shape = [600, 600]
    # 主干特征提取网络
    backbone = "resnet50"
    # 设置是否进行预训练
    pretrained = False

    # 设置锚点大小
    anchors_size = [8, 16, 32]

    Init_Epoch = 0
    Freeze_Epoch = 20
    Freeze_batch_size = 2
    Freeze_lr = 1e-4

    UnFreeze_Epoch = 30
    Unfreeze_batch_size = 1
    Unfreeze_lr = 1e-5

    Freeze_Train = True
    num_workers = 0

    #   获得图片路径和标签
    train_annotation_path = 'New_train.txt'
    val_annotation_path = 'New_val.txt'

    #   获取classes和anchor
    class_names, num_classes = get_classes(classes_path)

    model = FasterRCNN(num_classes, anchor_scales=anchors_size, backbone=backbone, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    # model.train() ： 让网络进入训练状
    model_train = model.train()
    if Cuda:
        # 当迭代次数或者epoch足够大的时候，我们通常会使用nn.DataParallel函数来用多个GPU来加速训练。
        model_train = torch.nn.DataParallel(model)
        # 会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
        cudnn.benchmark = True
        model_train = model_train.cuda()
    # 用于创建logs文件
    loss_history = LossHistory("logs/")

    #   读取数据集对应的txt
    with open(train_annotation_path) as f:
        # readlines() : 作为列表返回文件中的所有行，其中每一行都是列表对象中的一项：
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if True:
        batch_size = Freeze_batch_size
        # leaning rete -> lr
        lr = Freeze_lr
        # 初始训练
        start_epoch = Init_Epoch
        # 结束训练 --冻结阶段的轮数：20
        end_epoch = Freeze_Epoch

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        # 定义优化器
        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        # 每过step_size个epoch，做一次更新，调整学习率
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        # train_lines : 存储文件地址+位置+标签
        train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
        val_dataset = FRCNNDataset(val_lines, input_shape, train=False)
        # 载入数据集
        # shuffle : 在每个epoch开始的时候，对数据进行重新排序
        # drop_last : 如果设置为True：这个是对最后的未完成的batch来说的，比如你的batch_size设置为64，而一个epoch只有100个样本，
        # 那么训练的时候后面的36个就被扔掉了…
        # collate_fn : 将一个list的sample组成一个mini-batch的函数
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=frcnn_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=frcnn_dataset_collate)

        #   冻结一定部分训练
        if Freeze_Train:
            # 特征提取层 ： model.extractor
            for param in model.extractor.parameters():
                # 不需要为它计算梯度
                param.requires_grad = False

        #   冻结bn层
        model.freeze_bn()

        train_util = FasterRCNNTrainer(model, optimizer)

        for epoch in range(start_epoch, end_epoch):
            # 进入训练
            fit_one_epoch(model, train_util, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          end_epoch, Cuda)
            lr_scheduler.step()

    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
        val_dataset = FRCNNDataset(val_lines, input_shape, train=False)
        # 取完了一个批次, 然后进入self.collate_fn(data)进行整合，就得到了我们一个批次的data，最终我们返回来
        # Dataloader --> DataSet --> getitem()   取回image, box, label
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=frcnn_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=frcnn_dataset_collate)

        #   冻结一定部分训练
        if Freeze_Train:
            for param in model.extractor.parameters():
                param.requires_grad = True

        #   冻结bn层
        model.freeze_bn()

        train_util = FasterRCNNTrainer(model, optimizer)

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model, train_util, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          end_epoch, Cuda)
            lr_scheduler.step()
