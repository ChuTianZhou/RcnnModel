import math

import torch.nn as nn
from torch.hub import load_state_dict_from_url

# ResNet50有两个基本的块，分别名为Conv Block和Identity Block，
# 其中Conv Block输入和输出的维度是不一样的，所以不能连续串联，它的作用是改变网络的维度；Identity Block输入维度和输出维度相同，可以串联，用于加深网络的。
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # 进行三次卷积  分别是 1 3 1
        # conv2d: 输入通道数，输出通道数，卷积核
        # 首先进行 1*1 的卷积压缩通道数
        # 为什么 bias设置为False，因为一般下面进行BatchNorm2d时,设置它也不起作用，而且占用显存
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        # 进行归一化处理
        self.bn1 = nn.BatchNorm2d(planes)
        # 3*3的卷积 进行特征提取
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 再进行1*1的卷积 扩张通道数
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # 如果残差边有卷积的话，就会对残差边进行一个卷积
        if self.downsample is not None:
            residual = self.downsample(x)
        #  然后和输出 进行一个相加
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    # 此处的 num_classes = 1000 是什么意思？
    def __init__(self, block, layers, num_classes=1000):
        #-----------------------------------#
        #   假设输入进来的图片是600,600,3
        #-----------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()

        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 待处理的数据的通道数  out_channel,特征数量
        self.bn1 = nn.BatchNorm2d(64)
        # inplace=True  : 它会把输出直接覆盖到输入中，这样可以节省内存/显存
        self.relu = nn.ReLU(inplace=True)

        # 300,300,64 -> 150,150,64   我算的是 149.5  可能是向上取整
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        # 接下来 进行convBlock 和 identityBlock
        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 相当于依读取 网络模型，按深度优先遍历读取
        # 遍历self.modules()的每一层，然后判断当前层属于什么类型，是否是Conv2d，是否是BatchNorm2d，是否是Linear的，
        # 然后根据不同类型的层，设定不同的权值初始化方法
        for m in self.modules():
            # 判断两个类型是否相同？
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #-------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        #-------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 首先定义了这个残差边
            # 特征图尺寸减半：卷积步长stride=2。
            # 特征图通道加倍：卷积核数目out_channels=4*in_channels（因为H/2,W/2，特征图缩小为1/4，所以通道数x4）。
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 输入通道数 扩大4倍
        self.inplanes = planes * block.expansion
        # 进行1到 blocks次的 identityBlock
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        # 返回这一层的 所做的Sequential() 操作
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet50(pretrained = False):
    # 为什么是 3 4 6 3？  因为这对应了 Resnet50的结构 看那张图！ 第一步 有三个block    --第五次压缩的时候，有三个block：conv-->identity-->identity
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    # if pretrained:
    #     state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir="./model_data")
    #     model.load_state_dict(state_dict)
    #----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    #----------------------------------------------------------------------------#
    features    = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    #----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    #----------------------------------------------------------------------------#
    classifier  = list([model.layer4, model.avgpool])
    # 特征提取
    # Faster-RCNN的主干特征提取网络部分只包含了长宽压缩了四次的内容，第五次压缩后的内容在ROI中使用
    features    = nn.Sequential(*features)
    # 分类和回归预测
    classifier  = nn.Sequential(*classifier)
    return features, classifier
#
# if __name__ == '__main__':
#     extractor1, classifier2 = resnet50(pretrained=False)