import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from .cbam import *
# from cbam import *      # for local test
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    """
    input: (inplanes) channels -> output: (planes) channels

    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes, 16)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    """
    input: (inplanes) channels -> output: (planes * 4) channels
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes * 4, 16 )
        else:
            self.cbam = None

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

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, input_channel, layers,  network_type, num_classes, att_type=None,
                 flg_drop=False, r_drop=0.5):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        self.start_channel = input_channel
        self.model_name = ''
        # different model config between ImageNet and CIFAR
        self.flg_drop = False
        if flg_drop:
            self.flg_drop = True
        self.r_drop = r_drop


        kernel_size = 3
        self.conv1 = nn.Conv2d(input_channel, 64,
                               kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
        # avg pool size should be adjusted
        self.drop_0 = nn.Dropout2d(self.r_drop)
        self.avgpool = nn.AvgPool2d(kernel_size=6)

        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type)
        self.drop_1 = nn.Dropout2d(self.r_drop)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, att_type=att_type)
        self.drop_2 = nn.Dropout2d(self.r_drop)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
        self.drop_3 = nn.Dropout2d(self.r_drop)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type)
        self.drop_4 = nn.Dropout2d(self.r_drop)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x):
        ''''''
        x = self.conv1(x)

        x = self.bn1(x)
        # x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.flg_drop:
            x = self.drop_4(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = torch.sigmoid(x)

        return x


def ResidualNet_Clss(network_type, input_channel, depth, num_classes, att_type,
                     flg_drop=False, r_drop=0.5):

    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [6, 10, 18, 34, 50, 80], 'network depth should be 6, 10, 18, 34, 50 or 80'

    if depth == 18:
        model = ResNet(BasicBlock, input_channel, [2, 2, 2, 2], network_type, num_classes, att_type,
                       flg_drop=flg_drop, r_drop=r_drop)
        model.model_name = "ResidualNet_Clss_L18"

    if depth == 10:
        model = ResNet(BasicBlock, input_channel, [1, 1, 1, 1], network_type, num_classes, att_type,
                       flg_drop=flg_drop, r_drop=r_drop)
        model.model_name = "ResidualNet_Clss_L10"

    elif depth == 34:
        model = ResNet(BasicBlock, input_channel, [3, 4, 6, 3], network_type, num_classes, att_type,
                       flg_drop=flg_drop, r_drop=r_drop)
        model.model_name = "ResidualNet_Clss_L34"

    elif depth == 50:
        model = ResNet(BasicBlock, input_channel, [4, 5, 11, 4], network_type, num_classes, att_type,
                       flg_drop=flg_drop, r_drop=r_drop)
        model.model_name = "ResidualNet_Clss_L50"

    elif depth == 80:
        model = ResNet(BasicBlock, input_channel, [5, 6, 23, 5], network_type, num_classes, att_type,
                       flg_drop=flg_drop, r_drop=r_drop)
        model.model_name = "ResidualNet_Clss_L80"

    return model


def ResidualNet_Linr(network_type, input_channel, depth, att_type, flg_drop=False, r_drop=0.5):

    assert network_type in ["ImageNet"], "network type should be ImageNet"
    assert depth in [6, 10, 18, 34, 50, 80, 102], 'network depth should be 6, 10, 18, 34, 50 80 or 102'

    if depth == 18:
        model = ResNet(BasicBlock, input_channel, [2, 2, 2, 2], network_type, 1, att_type,
                       flg_drop=flg_drop, r_drop=r_drop)
        model.model_name = "ResidualNet_CBAM_Linr_L18"

    if depth == 10:
        model = ResNet(BasicBlock, input_channel, [1, 1, 1, 1], network_type, 1, att_type,
                       flg_drop=flg_drop, r_drop=r_drop)
        model.model_name = "ResidualNet_CBAM_Linr_L10"

    elif depth == 34:
        model = ResNet(BasicBlock, input_channel, [4, 4, 6, 3], network_type, 1, att_type,
                       flg_drop=flg_drop, r_drop=r_drop)
        model.model_name = "ResidualNet_CBAM_Linr_L34"

    elif depth == 50:
        model = ResNet(BasicBlock, input_channel, [4, 5, 11, 4], network_type, 1, att_type,
                       flg_drop=flg_drop, r_drop=r_drop)
        model.model_name = "ResidualNet_CBAM_Linr_L50"

    elif depth == 80:
        model = ResNet(BasicBlock, input_channel, [5, 6, 23, 5], network_type, 1, att_type)
        model.model_name = "ResidualNet_CBAM_Linr_L80"

    elif depth == 102:
        model = ResNet(BasicBlock, input_channel, [10, 10, 22, 8], network_type, 1, att_type)
        model.model_name = "ResidualNet_CBAM_Linr_L102"

    return model


''''''
if __name__ == '__main__':
    model = ResidualNet_Linr('ImageNet', 1, 50, 5, 'CBAM')
    model.eval()
    torch.set_grad_enabled(False)
    h_img = 24
    w_img = 24
    pc = np.zeros([1, 1, h_img, w_img])
    pc[0, 0, :, :] = np.random.randn(h_img, w_img)
    # pc[0, 1, :, :] = np.random.randn(h_img, w_img)
    # pc[0, 2, :, :] = np.random.randn(h_img, w_img)

    pc = torch.from_numpy(pc.astype(np.float32))

    label = model(pc)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    # b = np.copy(a[0])
    # print(b)

    print("END.")

