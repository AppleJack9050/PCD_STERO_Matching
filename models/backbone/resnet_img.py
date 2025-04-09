import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # Downsample if input and output dimensions differ
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNetFPN_4_1(nn.Module):
    """
    ResNet+FPN architecture that outputs:
      - A feature map at original resolution with 128 channels.
      - A feature map at 1/4 resolution with 256 channels.
      
    Config expects:
      - 'initial_dim': number of channels for the stem (e.g., 128)
      - 'block_dims': a list defining the channels for each block (e.g., [128, 196, 256])
    """
    def __init__(self, config):
        super().__init__()
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']  # e.g., [128, 196, 256]
        
        self.in_planes = initial_dim
        # Stem: outputs 1/2 resolution feature map
        self.conv1 = nn.Conv2d(3, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # Backbone layers
        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # remains 1/2 resolution
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # becomes 1/4 resolution
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # becomes 1/8 resolution
        
        # FPN branch for 1/4 output: fuse layer2 and upsampled layer3 features
        self.fpn_conv2 = conv1x1(block_dims[1], 256)
        self.fpn_conv3 = conv1x1(block_dims[2], 256)
        self.refine_2 = nn.Sequential(
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # FPN branch for original resolution output: upsample layer1 features (1/2 resolution) to original size
        self.fpn_conv1 = conv1x1(block_dims[0], 128)
        self.refine_1 = nn.Sequential(
            conv3x3(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block, planes, stride=1):
        layers = []
        layers.append(block(self.in_planes, planes, stride=stride))
        layers.append(block(planes, planes, stride=1))
        self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Stem: original input -> 1/2 resolution
        x0 = self.relu(self.bn1(self.conv1(x)))  # 1/2 resolution
        
        # Backbone
        x1 = self.layer1(x0)  # 1/2 resolution
        x2 = self.layer2(x1)  # 1/4 resolution
        x3 = self.layer3(x2)  # 1/8 resolution
        
        # FPN for 1/4 resolution: fuse x2 and upsampled x3
        x3_proj = self.fpn_conv3(x3)
        x3_upsampled = F.interpolate(x3_proj, scale_factor=2, mode='bilinear', align_corners=True)
        x2_fpn = self.fpn_conv2(x2) + x3_upsampled
        x2_fpn = self.refine_2(x2_fpn)
        
        # FPN for original resolution: upsample x1 from 1/2 to original size
        x1_proj = self.fpn_conv1(x1)
        x1_upsampled = F.interpolate(x1_proj, scale_factor=2, mode='bilinear', align_corners=True)
        x1_upsampled = self.refine_1(x1_upsampled)
        
        # Return a tuple:
        # - First element: original resolution feature map (dim 128)
        # - Second element: 1/4 resolution feature map (dim 256)
        return x1_upsampled, x2_fpn