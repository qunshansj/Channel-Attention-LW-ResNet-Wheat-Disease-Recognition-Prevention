python

class ECANet(nn.Module):
    def __init__(self, channels):
        super(ECANet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        G = 16  # group count
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.gconv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, groups=G, bias=False)
        self.gconv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, groups=G, padding=1, bias=False)
        self.gconv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=5, groups=G, padding=2, bias=False)
        self.gconv7 = nn.Conv2d(mid_channels, mid_channels, kernel_size=7, groups=G, padding=3, bias=False)

        self.conv2 = nn.Conv2d(mid_channels*4, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        g1 = self.gconv1(out)
        g3 = self.gconv3(out)
        g5 = self.gconv5(out)
        g7 = self.gconv7(out)
        out = torch.cat([g1, g3, g5, g7], dim=1)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class LWResNet(nn.Module):
    def __init__(self, num_classes=2):  # Assuming Binary Classification (Disease/Not Disease)
        super(LWResNet, self).__init__()

        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_layer(16, 16, 1)
        self.stage2 = self._make_layer(16, 32, 2)
        self.stage3 = self._make_layer(32, 64, 2)
        self.stage4 = self._make_layer(64, 128, 2)

        self.ecanet1 = ECANet(16)
        self.ecanet2 = ECANet(32)
        self.ecanet3 = ECANet(64)
        self.ecanet4 = ECANet(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, in_channels, out_channels, stride):
        return ResidualBlock(in_channels, out_channels, stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.ecanet1(x)

        x = self.stage2(x)
        x = self.ecanet2(x)

        x = self.stage3(x)
        x = self.ecanet3(x)

        ......



