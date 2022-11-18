import torch
from torch import nn
from torchvision import models
from torchvision.models.resnet import conv3x3

class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_convs=1, concat=True):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        num_channels_conv1 = in_channels + out_channels if concat else in_channels
        self.convs = nn.ModuleList([conv3x3(num_channels_conv1, out_channels, 1)])
        self.relu = nn.ReLU(inplace=True)
        for i in range(1, num_convs):
            self.convs.append(conv3x3(out_channels, out_channels, 1))

    def forward(self, x, x_prev=None):

        x = self.upsample(x)
        if x_prev is not None:
            x = torch.concat((x, x_prev), dim=1)
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.relu(x)
        return x


class HeatmapResNet18Model(torch.nn.Module):
    def __init__(self, num_keypoints, image_size=128):
        super(HeatmapResNet18Model, self).__init__()
        self.backbone = models.resnet18(pretrained=True)

        # Change output stride to 16
        self.backbone.layer4[0].conv1.stride = (1, 1)
        self.backbone.layer4[0].downsample[0].stride = (1, 1)
        for i in range(1, len(self.backbone.layer4)):
            cur_block = self.backbone.layer4[i]
            cur_block.conv2.dilation = (2, 2)
            cur_block.conv2.padding = (2, 2)

        self.upsample0 = UpsampleBlock(64, 64, num_convs=2, concat=False)
        self.upsample1 = UpsampleBlock(64, 64)
        self.upsample2 = UpsampleBlock(128, 64)
        self.upsample3 = UpsampleBlock(512, 128)
        #self.upsample4 = UpsampleBlock(512, 256)
        self.relu = nn.ReLU(inplace=True)
        self.final_cov = conv3x3(64, num_keypoints)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x1 = self.backbone.relu(self.backbone.bn1(x))
        x2 = self.backbone.maxpool(x1)
        x2 = self.backbone.layer1(x2)
        x3 = self.backbone.layer2(x2)
        x4 = self.backbone.layer3(x3)
        x = self.backbone.layer4(x4)
        #x = self.upsample4(x, x4)
        x = self.upsample3(x, x3)
        x = self.upsample2(x, x2)
        x = self.upsample1(x, x1)
        x = self.upsample0(x)
        x = self.final_cov(x)

        return x


class HeatmapResNet50Model(torch.nn.Module):
    def __init__(self, num_keypoints, image_size=128):
        super(HeatmapResNet50Model, self).__init__()

        self.backbone = models.resnet50(pretrained=True)

        # Change output stride to 16
        self.backbone.layer4[0].conv2.stride = (1, 1)
        self.backbone.layer4[0].downsample[0].stride = (1, 1)
        for i in range(1, len(self.backbone.layer4)):
            cur_block = self.backbone.layer4[i]
            cur_block.conv2.dilation = (2, 2)
            cur_block.conv2.padding = (2, 2)

        self.upsample0 = UpsampleBlock(64, 64, num_convs=2, concat=False)
        self.upsample1 = UpsampleBlock(256, 64, num_convs=2)
        self.upsample2 = UpsampleBlock(512, 256, num_convs=2)
        self.upsample3 = UpsampleBlock(2048, 512, num_convs=2)
        #self.upsample4 = UpsampleBlock(2048, 1024, num_convs=2)
        self.relu = nn.ReLU(inplace=True)
        self.final_cov = conv3x3(64, num_keypoints)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x1 = self.backbone.relu(self.backbone.bn1(x))
        x2 = self.backbone.maxpool(x1)
        x2 = self.backbone.layer1(x2)
        x3 = self.backbone.layer2(x2)
        x4 = self.backbone.layer3(x3)
        x = self.backbone.layer4(x4)
        #x = self.upsample4(x, x4)  # Due to output stride = 16
        x = self.upsample3(x, x3)
        x = self.upsample2(x, x2)
        x = self.upsample1(x, x1)
        x = self.upsample0(x)
        x = self.final_cov(x)

        return x

class HeatmapResNet101Model(torch.nn.Module):
    def __init__(self, num_keypoints, image_size=128):
        super(HeatmapResNet101Model, self).__init__()

        self.backbone = models.resnet101(pretrained=True)

        # Change output stride to 16
        self.backbone.layer4[0].conv2.stride = (1, 1)
        self.backbone.layer4[0].downsample[0].stride = (1, 1)
        for i in range(1, len(self.backbone.layer4)):
            cur_block = self.backbone.layer4[i]
            cur_block.conv2.dilation = (2, 2)
            cur_block.conv2.padding = (2, 2)

        self.upsample0 = UpsampleBlock(64, 64, num_convs=2, concat=False)
        self.upsample1 = UpsampleBlock(256, 64, num_convs=2)
        self.upsample2 = UpsampleBlock(512, 256, num_convs=2)
        self.upsample3 = UpsampleBlock(2048, 512, num_convs=2)
        #self.upsample4 = UpsampleBlock(2048, 1024, num_convs=2)
        self.relu = nn.ReLU(inplace=True)
        self.final_cov = conv3x3(64, num_keypoints)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x1 = self.backbone.relu(self.backbone.bn1(x))
        x2 = self.backbone.maxpool(x1)
        x2 = self.backbone.layer1(x2)
        x3 = self.backbone.layer2(x2)
        x4 = self.backbone.layer3(x3)
        x = self.backbone.layer4(x4)
        #x = self.upsample4(x, x4)  # Due to output stride = 16
        x = self.upsample3(x, x3)
        x = self.upsample2(x, x2)
        x = self.upsample1(x, x1)
        x = self.upsample0(x)
        x = self.final_cov(x)

        return x
