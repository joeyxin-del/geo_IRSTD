from torch import nn
import torch
import torch.nn.functional as F

from .Blocks import *

class Stage(nn.Sequential):
    """Stage containing multiple BasicBlocks."""
    def __init__(self, block, num_blocks, in_channels, out_channels, stride=1):
        layers = []
        # First block with downsampling
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers.append(block(in_channels, out_channels, stride=stride, downsample=downsample))
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        super(Stage, self).__init__(*layers)

class NoiseAttNet(nn.Module):
    def __init__(self):
        super(NoiseAttNet, self).__init__()

        self.depths = [2,2,2,2]
        # self.Blocks = [BasicBlock,BasicBlock,BasicBlock,BasicBlock]
        self.Blocks = [BasicBlock, BasicBlock, BasicBlock, linearAttBlock]
        # self.Blocks = [BasicBlock, BasicBlock, linearAttBlock, BasicBlock]
        # self.Blocks = [BasicBlock, linearAttBlock, BasicBlock, BasicBlock]
        # self.Blocks = [linearAttBlock, BasicBlock, BasicBlock, BasicBlock]
        self.channels = [16, 32, 64, 128, 256]


        # stemlayer (1,1,512,512) -> (1,16,256,256)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channels[0], kernel_size=3, stride=2, padding=1),
            # norm
            nn.BatchNorm2d(num_features=self.channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # backbone stage1 (1, 16, 256, 256) -> (1, 32, 128, 128)
        # backbone stage2 (1, 32, 128, 128) -> (1, 64, 64, 64)
        # backbone stage3 (1, 64, 64, 64) -> (1, 128, 32, 32)
        # backbone stage4 (1, 128, 32, 32) -> (1, 256, 16, 16)

        self.stage_0 = Stage(self.Blocks[0], self.depths[0], self.channels[0], self.channels[1], stride=1)
        self.stage_1 = Stage(self.Blocks[1], self.depths[1], self.channels[1], self.channels[2], stride=2)
        self.stage_2 = Stage(self.Blocks[2], self.depths[2], self.channels[2], self.channels[3], stride=2)
        self.stage_3 = Stage(self.Blocks[3], self.depths[3], self.channels[3], self.channels[4], stride=2)



        # fusion_add_3 up(conv1(backbone_layer4))+conv1(backbone_layer3) -> (1, 64, 32, 32)
        # fusion_add_2 up(fusion_add3)+conv1(backbone_layer2) -> (1, 64, 64, 64)
        # self.lateral_conv_4 = nn.Conv2d(in_channels=self.channels[4], out_channels=self.channels[2], kernel_size=1)
        # self.lateral_conv_3 = nn.Conv2d(in_channels=self.channels[3], out_channels=self.channels[2], kernel_size=1)
        self.lateral_conv_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels[4],
                out_channels=self.channels[2],
                kernel_size=1,
                stride=1
            ),
            nn.BatchNorm2d(self.channels[2]),
            nn.ReLU(inplace=True)
        )
        self.lateral_conv_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels[3],
                out_channels=self.channels[2],
                kernel_size=1,
                stride=1
            ),
            nn.BatchNorm2d(self.channels[2]),
            nn.ReLU(inplace=True)
        )


        # self.out_FPN_conv3 = nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[2], kernel_size=3, padding=1)
        # self.out_FPN_conv2 = nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[2], kernel_size=3, padding=1)
        self.out_FPN_conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels[2],
                out_channels=self.channels[2],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(self.channels[2]),
            nn.ReLU(inplace=True)
        )
        self.out_FPN_conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels[2],
                out_channels=self.channels[2],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(self.channels[2]),
            nn.ReLU(inplace=True)
        )


        # fusion_concat_3 conv1(up(conv3(fusion_add_3)) concat conv3(fusion_add_2)) -> (1, 64, 64, 64)
        # fusion_concat_2 conv1(up(conv3(fusion_concat_3)) concat conv3(backbone_layer1)) -> (1, 32, 128, 128)
        # self.fusion_concat_2 = nn.Conv2d(self.channels[2] * 2, self.channels[2], kernel_size=1)
        # self.fusion_concat_1 = nn.Conv2d(self.channels[2] + self.channels[1], self.channels[1], kernel_size=1)
        self.fusion_concat_2 =nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels[2] * 2,
                out_channels=self.channels[2],
                kernel_size=1,
                stride=1
            ),
            nn.BatchNorm2d(self.channels[2]),
            nn.ReLU(inplace=True)
        )
        self.fusion_concat_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels[2] + self.channels[1],
                out_channels=self.channels[1],
                kernel_size=1,
                stride=1
            ),
            nn.BatchNorm2d(self.channels[1]),
            nn.ReLU(inplace=True)
        )
        # head_layer1 (1, 32, 128, 128) -> (1, 16, 128,128)
        # head_layer2 (1, 16, 128, 128) -> (1, 8, 256, 256)
        # head_layer3 (1, 8, 256, 256) -> (1, 4, 512, 512)
        # head_layer4 (1, 4, 512, 512) -> (1, 1, 512, 512)
        self.head_layer1 = nn.Conv2d(self.channels[1], self.channels[0], kernel_size=3, padding=1)
        self.head_layer2 = nn.ConvTranspose2d(self.channels[0], 8, kernel_size=4, stride=2, padding=1)
        self.head_layer3 = nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1)
        self.head_layer4 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Stem
        x0 = self.stem(x)

        # Backbone
        x1 = self.stage_0(x0)
        x2 = self.stage_1(x1)
        x3 = self.stage_2(x2)
        x4 = self.stage_3(x3)
        # print(" x1 shape",x1.shape)
        # print(" x2 shape",x2.shape)
        # print(" x3 shape",x3.shape)
        # print(" x4 shape",x4.shape)

        # Fusion
        up3 = F.interpolate(self.lateral_conv_4(x4), size=x3.shape[2:], mode='bilinear', align_corners=False) + self.lateral_conv_3(x3)
        up2 = F.interpolate(up3, size=x2.shape[2:], mode='bilinear', align_corners=False) + x2
        # print(" up3 shape",up3.shape)
        # print(" up2 shape",up2.shape)

        up3 = self.out_FPN_conv3(up3)
        up2 = self.out_FPN_conv3(up2)

        concat2 = self.fusion_concat_2(
            torch.cat([F.interpolate(up3, size=x2.shape[2:], mode='bilinear', align_corners=False), up2], dim=1))
        concat1 = self.fusion_concat_1(
            torch.cat([F.interpolate(concat2, size=x1.shape[2:], mode='bilinear', align_corners=False), x1], dim=1))

        # print(" concat2 shape",concat2.shape)
        # print(" concat1 shape",concat1.shape)
        # Head layers
        h1 = self.head_layer1(concat1)
        h2 = self.head_layer2(h1)
        h3 = self.head_layer3(h2)
        h4 = self.head_layer4(h3)
        # print(" h1 shape",h1.shape)
        # print(" h2 shape",h2.shape)
        # print(" h3 shape",h3.shape)
        # print(" h4 shape",h4.shape)


        return h4.sigmoid()



if __name__ == "__main__":
    # Instantiate the model
    model = NoiseAttNet()

    # Create a dummy input tensor with the shape (batch_size=1, channels=1, height=512, width=512)
    dummy_input = torch.randn(1, 1, 512, 512)

    # Forward pass
    output = model(dummy_input)

    # Print the output shape
    print("Output shape:", output.shape)