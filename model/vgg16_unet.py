from torch import nn
from torch.nn.functional import sigmoid
from model.layers import (DouConvBlock,
                          TriConvBlock,
                          EncoderLayer,
                          DecoderLayer)


class VGG16Unet(nn.Module):
    def __init__(self):
        super(VGG16Unet, self).__init__()
        self.EnBlock1 = DouConvBlock(3, 64)
        self.EnBlock2 = DouConvBlock(64, 128)
        self.EnBlock3 = TriConvBlock(128, 256)
        self.EnBlock4 = TriConvBlock(256, 512)

        self.CenterBlock5 = TriConvBlock(512, 512)

        self.DeBlock1 = TriConvBlock(512, 512)
        self.DeBlock2 = TriConvBlock(256, 256)
        self.DeBlock3 = DouConvBlock(128, 128)
        self.DeBlock4 = DouConvBlock(64, 64)

        self.OutputLayer = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, tensor):
        skip1, pool1 = EncoderLayer(conv_block=self.EnBlock1)(tensor)
        skip2, pool2 = EncoderLayer(conv_block=self.EnBlock2)(pool1)
        skip3, pool3 = EncoderLayer(conv_block=self.EnBlock3)(pool2)
        skip4, pool4 = EncoderLayer(conv_block=self.EnBlock4)(pool3)

        pool5 = self.CenterBlock5(pool4)

        cat1 = (DecoderLayer(conv_block=self.DeBlock1, in_channel=512,
                             out_channel=512)(skip4, pool5))
        cat2 = DecoderLayer(conv_block=self.DeBlock2, in_channel=512,
                            out_channel=256)(skip3, cat1)
        cat3 = DecoderLayer(conv_block=self.DeBlock3, in_channel=256,
                            out_channel=128)(skip2, cat2)
        cat4 = DecoderLayer(conv_block=self.DeBlock4, in_channel=128,
                            out_channel=64)(skip1, cat3)

        out = self.OutputLayer(cat4)

        return sigmoid(out)
