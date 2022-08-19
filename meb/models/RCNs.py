import torch
import torch.nn as nn

__all__ = ['RCN_A', 'RCN_S', 'RCN_W', 'RCN_P', 'RCN_C', 'RCN_F']


class ConvBlock(nn.Module):
    """convolutional layer blocks for sequtial convolution operations"""
    def __init__(self, in_features, out_features, num_conv, pool=False):
        super(ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features)-1):
            layers.append(nn.Conv2d(in_channels=features[i], out_channels=features[i+1], kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i+1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)


class RclBlock(nn.Module):
    """recurrent convolutional blocks"""
    def __init__(self, inplanes, planes):
        super(RclBlock, self).__init__()
        self.ffconv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.rrconv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout()
        )

    def forward(self, x):
        y = self.ffconv(x)
        y = self.rrconv(x + y)
        y = self.rrconv(x + y)
        out = self.downsample (y)
        return out


class DenseBlock(nn.Module):
    """densely connected convolutional blocks"""
    def __init__(self, inplanes, planes):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout()
        )

    def forward(self, x):
        y = self.conv1(x)
        z = self.conv2(x + y)
        # out = self.conv2(x + y + z)
        e = self.conv2(x + y + z)
        out = self.conv2(x + y + z + e)
        out = self.downsample (out)
        return out


class EmbeddingBlock(nn.Module):
    """densely connected convolutional blocks for embedding"""
    def __init__(self, inplanes, planes):
        super(EmbeddingBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.attenmap = SpatialAttentionBlock_P(normalize_attn=True)
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout()
        )

    def forward(self, x, w, pool_size, classes):
        y = self.conv1(x)
        y1 = self.attenmap(F.adaptive_avg_pool2d(x, (pool_size, pool_size)), w, classes)
        y = torch.mul(F.interpolate(y1, (y.shape[2], y.shape[3])), y)
        z = self.conv2(x+y)
        e = self.conv2(x + y + z)
        out = self.conv2(x + y + z + e)
        out = self.downsample (out)
        return out


class EmbeddingBlock_M(nn.Module):
    """densely connected convolutional blocks for embedding with multiple attentions"""
    def __init__(self, inplanes, planes):
        super(EmbeddingBlock_M, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.attenmap = SpatialAttentionBlock_P(normalize_attn=True)
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout()
        )

    def forward(self, x, w, pool_size, classes):
        y = self.conv1(x)
        z = self.conv2(x + y)
        y1 = self.attenmap(F.adaptive_avg_pool2d(x, (pool_size, pool_size)), w, classes)
        z = torch.mul(F.interpolate(y1, (z.shape[2], z.shape[3])), z)
        e = self.conv2(x + y + z)
        out = self.conv2(x + y + z + e)
        y2 = self.attenmap(F.adaptive_avg_pool2d(z, (pool_size, pool_size)), w, classes)
        out = torch.mul(F.interpolate(y2, (out.shape[2], out.shape[3])), out)
        out = self.downsample (out)
        return out


class SpatialAttentionBlock_A(nn.Module):
    """linear attention block for any layers"""
    def __init__(self, in_features, normalize_attn=True):
        super(SpatialAttentionBlock_A, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)

    def forward(self, l):
        N, C, W, H = l.size()
        c = self.op(l) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        return g


class SpatialAttentionBlock_P(nn.Module):
    """linear attention block for any layers"""
    def __init__(self, normalize_attn=True):
        super(SpatialAttentionBlock_P, self).__init__()
        self.normalize_attn = normalize_attn

    def forward(self, l, w, classes):
        output_cam = []
        for idx in range(0,classes):
            weights = w[idx,:].reshape((l.shape[1], l.shape[2], l.shape[3]))
            cam = weights * l
            cam = cam.mean(dim=1,keepdim=True)
            cam = cam - torch.min(torch.min(cam,3,True)[0],2,True)[0]
            cam = cam / torch.max(torch.max(cam,3,True)[0],2,True)[0]
            output_cam.append(cam)
        output = torch.cat(output_cam, dim=1)
        output = output.mean(dim=1,keepdim=True)
        return output


class SpatialAttentionBlock_F(nn.Module):
    """linear attention block for first layer"""
    def __init__(self, normalize_attn=True):
        super(SpatialAttentionBlock_F, self).__init__()
        self.normalize_attn = normalize_attn

    def forward(self, l, w, classes):
        output_cam = []
        for idx in range(0,classes):
            weights = w[idx,:].reshape((-1, l.shape[2], l.shape[3]))
            weights = weights.mean(dim=0,keepdim=True)
            cam = weights * l
            cam = cam.mean(dim=1,keepdim=True)
            cam = cam - torch.min(torch.min(cam,3,True)[0],2,True)[0]
            cam = cam / torch.max(torch.max(cam,3,True)[0],2,True)[0]
            output_cam.append(cam)
        output = torch.cat(output_cam, dim=1)
        output = output.mean(dim=1,keepdim=True)
        return output


def MakeLayer(block, planes, blocks):
    layers = []
    for _ in range(0, blocks):
        layers.append(block(planes, planes))
    return nn.Sequential(*layers)

class RCN_A(nn.Module):
    """menet networks with adding attention unit
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5, model_version=3, **kwargs):
        super(RCN_A, self).__init__()
        self.version = model_version
        self.classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, featuremaps, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(featuremaps),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(),
        )
        self.rcls = MakeLayer(RclBlock, featuremaps, num_layers)
        self.attenmap = SpatialAttentionBlock_P(normalize_attn=True)
        self.downsampling = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.fc = nn.Linear(pool_size*pool_size*featuremaps, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.version == 1:
            x = self.conv1(x)
            x = self.attenmap(x)
            x = self.rcls(x)
            x = self.avgpool(x)
        if self.version == 2:
            x = self.conv1(x)
            x = self.attenmap(x)
            x = self.rcls(x)
            x = self.avgpool(x)
        elif self.version == 3:
            x = self.conv1(x)
            y = self.attenmap(self.downsampling(x), self.classifier.weight, self.classes)
            x = self.rcls(x)
            x = self.avgpool(x)
            x = x * y
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class RCN_S(nn.Module):
    """menet networks with dense shortcut connection
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5, **kwargs):
        super(RCN_S, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input, featuremaps, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(featuremaps),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(),
        )
        self.dbl = MakeLayer(DenseBlock, featuremaps, num_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Linear(pool_size*pool_size*featuremaps, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dbl(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class RCN_W(nn.Module):
    """menet networks with wide expansion
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5, **kwargs):
        super(RCN_W, self).__init__()
        num_channels = int(featuremaps/2)
        self.stream1 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream2 = nn.Sequential(
            # nn.Conv2d(num_input, num_channels, kernel_size=5, stride=3, padding=2),
            nn.Conv2d(num_input, int(num_channels/2), kernel_size=3, stride=3, padding=2, dilation=2),  # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(num_channels/2)),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream3 = nn.Sequential(
            # nn.Conv2d(num_input, num_channels, kernel_size=5, stride=3, padding=2),
            nn.Conv2d(num_input, int(num_channels/2), kernel_size=3, stride=3, padding=3, dilation=3),  # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(num_channels/2)),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.rcls = MakeLayer(RclBlock, featuremaps, num_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.fc = nn.Linear(pool_size*pool_size*featuremaps, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.stream1(x)
        x2 = self.stream2(x)
        x3 = self.stream3(x)
        x = torch.cat((x1,x2,x3),1)
        x = self.rcls(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class RCN_P(nn.Module):
    """menet networks with hybrid modules by NAS
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5, **kwargs):
        super(RCN_P, self).__init__()
        self.classes = num_classes
        num_channels = int(featuremaps/2)
        self.stream1 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=1, padding=1), # 1->3
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream2 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=1, padding=3, dilation=3), # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.dbl = MakeLayer(DenseBlock, featuremaps, num_layers)
        self.rcls = MakeLayer(RclBlock, featuremaps, num_layers)
        self.attenmap = SpatialAttentionBlock_P(normalize_attn=True)
        self.downsampling = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.fc = nn.Linear(pool_size*pool_size*featuremaps, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.stream1(x)
        x2 = self.stream2(x)
        x = torch.cat((x1,x2),1)
        y = self.attenmap(self.downsampling(x), self.classifier.weight, self.classes)
        x = self.dbl(x)
        x = self.avgpool(x)
        x = x * y
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class RCN_C(nn.Module):
    """menet networks with cascaded modules
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5, **kwargs):
        super(RCN_C, self).__init__()
        self.classes = num_classes
        self.poolsize = pool_size
        num_channels = int(featuremaps/2)
        # self.stream1 = nn.Sequential(
        #     nn.Conv2d(num_input, num_channels, kernel_size=3, stride=1, padding=1), # 1->3
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(num_channels),
        #     nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
        #     nn.Dropout(),
        # )
        # self.stream2 = nn.Sequential(
        #     nn.Conv2d(num_input, num_channels, kernel_size=5, stride=1, padding=2), # 5,2/ 1,0
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(num_channels),
        #     nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
        #     nn.Dropout(),
        # )
        self.stream1 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream2 = nn.Sequential(
            nn.Conv2d(num_input, int(num_channels/2), kernel_size=3, stride=3, padding=2, dilation=2),  # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(num_channels/2)),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream3 = nn.Sequential(
            nn.Conv2d(num_input, int(num_channels/2), kernel_size=3, stride=3, padding=3, dilation=3),  # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(num_channels/2)),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.dbl = MakeLayer(DenseBlock, featuremaps, num_layers)
        self.ebl = EmbeddingBlock(featuremaps, featuremaps)
        # self.attenmap = SpatialAttentionBlock_P(normalize_attn=True)
        self.attenmap = SpatialAttentionBlock_F(normalize_attn=True)
        self.downsampling = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.fcr = nn.Linear(pool_size*pool_size*featuremaps, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.attenmap(self.downsampling(x), self.classifier.weight, self.classes)
        x1 = self.stream1(x)
        x2 = self.stream2(x)
        x3 = self.stream3(x)
        x = torch.cat((x1, x2, x3), 1)
        # x = torch.cat((x1,x2),1)
        # y = self.attenmap(self.downsampling(x), self.classifier.weight, self.classes)
        x = torch.mul(F.interpolate(y,(x.shape[2],x.shape[3])), x)
        x = self.dbl(x)
        # x = self.ebl(x, self.classifier.weight, self.poolsize, self.classes)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class RCN_F(nn.Module):
    """menet networks with embedded modules as final fusion way
    """
    def __init__(self, num_input, featuremaps, num_classes, num_layers=1, pool_size=5, **kwargs):
        super(RCN_F, self).__init__()
        self.classes = num_classes
        self.poolsize = pool_size
        num_channels = int(featuremaps/2)
        # self.stream1 = nn.Sequential(
        #     nn.Conv2d(num_input, num_channels, kernel_size=3, stride=1, padding=1), # 1->3
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(num_channels),
        #     nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
        #     nn.Dropout(),
        # )
        # self.stream2 = nn.Sequential(
        #     nn.Conv2d(num_input, num_channels, kernel_size=5, stride=1, padding=2), # 5,2/ 1,0
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(num_channels),
        #     nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
        #     nn.Dropout(),
        # )
        self.stream1 = nn.Sequential(
            nn.Conv2d(num_input, num_channels, kernel_size=3, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_channels),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream2 = nn.Sequential(
            nn.Conv2d(num_input, int(num_channels/2), kernel_size=3, stride=3, padding=2, dilation=2),  # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(num_channels/2)),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.stream3 = nn.Sequential(
            nn.Conv2d(num_input, int(num_channels/2), kernel_size=3, stride=3, padding=3, dilation=3),  # 5,2/ 1,0
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(int(num_channels/2)),
            # nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
            nn.Dropout(),
        )
        self.ebl = EmbeddingBlock(featuremaps, featuremaps)
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.fc = nn.Linear(pool_size*pool_size*featuremaps, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.stream1(x)
        x2 = self.stream2(x)
        # x = torch.cat((x1, x2), 1)
        x3 = self.stream2(x)
        x = torch.cat((x1,x2,x3),1)
        x = self.ebl(x, self.classifier.weight, self.poolsize, self.classes)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x