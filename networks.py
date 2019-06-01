import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ConvBlock(nn.Module):
    def __init__(self, block_name, in_size, out_size, normalize=True, kernel_size=4, stride=2, padding=1, bias=False,
                 activation_fn=nn.LeakyReLU(0.2)):
        super(ConvBlock, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module(block_name + "_conv2d",
                              nn.Conv2d(in_size, out_size, kernel_size=kernel_size,
                                        stride=stride, padding=padding, bias=bias))
        if normalize:
            self.model.add_module(block_name + "_norm", nn.BatchNorm2d(out_size))
        if activation_fn is not None:
            self.model.add_module(block_name + "_activation_fn", activation_fn)

    def forward(self, x):
        x = self.model(x)
        return x


class TransConvBlock(nn.Module):
    def __init__(self, block_name, in_size, out_size, normalize=True,
                 kernel_size=4, stride=2, padding=1, bias=False, activation_fn=nn.ReLU()):
        super(TransConvBlock, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module(block_name + "_trans_conv",
                              nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size,
                                                 stride=stride, padding=padding, bias=bias))
        if normalize:
            self.model.add_module(block_name + "_norm", nn.BatchNorm2d(out_size))
        if activation_fn is not None:
            self.model.add_module(block_name + "_activation_fn", activation_fn)

    def forward(self, x):
        x = self.model(x)
        return x


# -------------------
# ---- Generator ----
# -------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.upconv1 = TransConvBlock("gen_start_block", 100, 1024, normalize=True, kernel_size=4, stride=1, padding=0)
        self.upconv2 = TransConvBlock("gen_mid_block1", 1024, 512, normalize=True, kernel_size=4, stride=2, padding=1)
        self.upconv3 = TransConvBlock("gen_mid_block2", 512, 256, normalize=True, kernel_size=4, stride=2, padding=1)
        self.upconv4 = TransConvBlock("gen_mid_block3", 256, 128, normalize=True, kernel_size=4, stride=2, padding=1)
        self.upconv5 = TransConvBlock("gen_end_block", 128, 3, normalize=False, kernel_size=4, stride=2, padding=1,
                                      activation_fn=nn.Tanh())

    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        return x


# -------------------------
# -- DCGAN Discriminator --
# -------------------------
class DCGAN_Discriminator(nn.Module):
    def __init__(self):
        super(DCGAN_Discriminator, self).__init__()

        self.conv1 = ConvBlock("disc_start_block", 3, 128, normalize=False, kernel_size=4, stride=2, padding=1)
        self.conv2 = ConvBlock("disc_mid_block1", 128, 256, normalize=True, kernel_size=4, stride=2, padding=1)
        self.conv3 = ConvBlock("disc_mid_block2", 256, 512, normalize=True, kernel_size=4, stride=2, padding=1)
        self.conv4 = ConvBlock("disc_mid_block3", 512, 1024, normalize=True, kernel_size=4, stride=2, padding=1)
        self.conv5 = ConvBlock("disc_end_block", 1024, 1, normalize=False, kernel_size=4, stride=1, padding=0,
                               activation_fn=nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1)
        return x


# ------------------------
# -- WGAN Discriminator --
# ------------------------
class WGAN_Discriminator(nn.Module):
    def __init__(self):
        super(WGAN_Discriminator, self).__init__()

        self.conv1 = ConvBlock("disc_start_block", 3, 128, normalize=False, kernel_size=4, stride=2, padding=1)
        self.conv2 = ConvBlock("disc_mid_block1", 128, 256, normalize=False, kernel_size=4, stride=2, padding=1)
        self.conv3 = ConvBlock("disc_mid_block2", 256, 512, normalize=False, kernel_size=4, stride=2, padding=1)
        self.conv4 = ConvBlock("disc_mid_block3", 512, 1024, normalize=False, kernel_size=4, stride=2, padding=1)
        self.conv5 = ConvBlock("disc_end_block", 1024, 1, normalize=False, kernel_size=4, stride=1, padding=0,
                               activation_fn=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.mean(0)
        x = x.view(1)
        return x
