from thop import profile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from EMFFTrans.EMFFTrans import EMFFTrans




class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch,num_block):
        super(down, self).__init__()
        
        self.LMCB = nn.Sequential()
        self.LMCB.add_module('MaxPool2D', nn.MaxPool2d(2))
        self.LMCB.add_module("LMCBModule_D_0",
                                    LMCBModuleD(in_ch,out_ch, d=2))
        for i in range(1, num_block):
            self.LMCB.add_module("LMCBModule_D_" + str(i),
                                        LMCBModuleD(out_ch,out_ch, d=2))

    def forward(self, x):
        x = self.LMCB(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        self.bilinear = bilinear

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.lmcb = nn.Sequential(LMCBModuleD(in_ch,out_ch, d=2),
                                  LMCBModuleD(out_ch,out_ch, d=2))

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)


        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])



        x = torch.cat([x2, x1], dim=1)
        x = self.lmcb(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.pa = PA(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.pa(x)
        x = self.conv(x)
        return x
    
    
    
    

class PA(nn.Module):

    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out




class EMFFTransNet(nn.Module):
    def __init__(self, classes):
        super(EMFFTransNet, self).__init__()
        self.inc = inconv(3, 16)
        self.down1 = down(16, 16,2)
        self.down2 = down(16, 32,4)
        self.down3 = down(32, 64,6)
        self.down4 = down(64, 128,8)

        
        self.transformer = EMFFTrans( vis = False, img_h=360,img_w=480, embeddings_dropout_rate = 0, num_layers=2,
                                      channel_num=[16, 32, 64, 128],
                                      n_patch_h = 4, n_patch_w = 4)
        self.up1 = up(192, 64)
        self.up2 = up(96, 32)
        self.up3 = up(48, 32)
        self.up4 = up(48, 16)
        self.outc = outconv(16, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        b, c, h, w = x5.shape



        x2,x3,x4,x5,att_weights = self.transformer(x2,x3,x4,x5)
        
        
        x = self.up1(x5, x4)

        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)


        return x
    def resize_output_if_needed(self, x,orig_h,orig_w):

        x = F.interpolate(
            x, size=(orig_h, orig_w), mode="bilinear", align_corners=True
        )

        return x


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output



class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output
    
    
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()


        y = self.avg_pool(x)

        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)


        y = self.sigmoid(y)

        return x * y.expand_as(x)


    





class LMCBModuleA(nn.Module):
    def __init__(self, nIn,nOut, d=2, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv1x1_in = Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=False)
        self.depth_conv = Conv(nIn // 2, nIn // 2, (dkSize, dkSize), 1, padding=(1, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=1, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=1, bn_acti=True)
        self.point_conv = Conv(nIn // 2, nIn//2, 1, 1, padding=0, groups=1, bn_acti=True)
  
        if nIn == nOut:
            self.ca = eca_layer(nIn)
        else:
            self.ca = nn.Sequential(Conv(nIn,nOut, 1, 1, padding=0, bn_acti=False),
                                     eca_layer(nOut))
            
        self.bn_relu_2 = BNPReLU(nOut)
        self.conv1x1_out = Conv(nIn//2, nOut, 1, 1, padding=0, bn_acti=False)        
        self.shuffle = ShuffleBlock(nOut)


    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_in(output)
        output = self.depth_conv(output)
        output = self.point_conv(output)
        
        
        output = self.ddconv3x1(output)
        output = self.ddconv1x3(output)

        output = self.conv1x1_out(output)

        output = self.bn_relu_2(output)

        return output







class LMCBModuleB(nn.Module):
    def __init__(self, nIn,nOut, d=2, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv1x1_in = Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=False)
        self.depth_conv = Conv(nIn // 2, nIn // 2, (dkSize, dkSize), 1, padding=(1, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=1, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=1, bn_acti=True)
        self.point_conv = Conv(nIn // 2, nIn//2, 1, 1, padding=0, groups=1, bn_acti=True)
  
        if nIn == nOut:
            self.ca = eca_layer(nIn)
        else:
            self.ca = nn.Sequential(Conv(nIn,nOut, 1, 1, padding=0, bn_acti=False),
                                     eca_layer(nOut))
            
        self.bn_relu_2 = BNPReLU(nOut)
        self.conv1x1_out = Conv(nIn//2, nOut, 1, 1, padding=0, bn_acti=False)        
        self.shuffle = ShuffleBlock(nOut)


    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_in(output)
        x1 = self.depth_conv(output)
        x1 = self.point_conv(output)
        
        
        x2 = self.ddconv3x1(output)
        x2 = self.ddconv1x3(output)
        out = x1 + x2
        output = self.conv1x1_out(output)

        output = self.bn_relu_2(output)

        return output














class LMCBModuleC(nn.Module):
    def __init__(self, nIn,nOut, d=2, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv1x1_in = Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=False)
        self.depth_conv = Conv(nIn // 2, nIn // 2, (dkSize, dkSize), 1, padding=(1, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.point_conv = Conv(nIn // 2, nIn//2, 1, 1, padding=0, groups=1, bn_acti=True)
        if nIn == nOut:
            self.ca = eca_layer(nIn)
        else:
            self.ca = nn.Sequential(Conv(nIn,nOut, 1, 1, padding=0, bn_acti=False),
                                     eca_layer(nOut))
        self.bn_relu_2 = BNPReLU(nOut)
        self.conv1x1_out = Conv(nIn//2, nOut, 1, 1, padding=0, bn_acti=False)        
        self.shuffle = ShuffleBlock(nOut)


    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_in(output)
        output = self.depth_conv(output)
        output = self.ddconv3x1(output)
        output = self.ddconv1x3(output)
        output = self.point_conv(output)
        output = self.conv1x1_out(output)

        output = self.bn_relu_2(output)

        return output
        
    
    
    
    

class LMCBModuleD(nn.Module):
    def __init__(self, nIn,nOut, d=2, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv1x1_in = Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=False)
        self.depth_conv = Conv(nIn // 2, nIn // 2, (dkSize, dkSize), 1, padding=(1, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.point_conv = Conv(nIn // 2, nIn // 2, 1, 1, padding=0, groups=1, bn_acti=True)
        if nIn == nOut:
            self.ca = eca_layer(nIn)
        else:
            self.ca = nn.Sequential(Conv(nIn,nOut, 1, 1, padding=0, bn_acti=False),
                                     eca_layer(nOut))
        self.bn_relu_2 = BNPReLU(nOut)
        self.conv1x1_out = Conv(nIn // 2, nOut, 1, 1, padding=0, bn_acti=False)        
        self.shuffle = ShuffleBlock(nOut)


    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_in(output)
        x1 = self.depth_conv(output)
        x2 = self.ddconv3x1(output)
        x2 = self.ddconv1x3(x2)
        out = x1 + x2
        output = self.point_conv(output)
        output = self.conv1x1_out(output)
        att = self.ca(input)
        
        output = att + output
        output = self.bn_relu_2(output)

        output = self.shuffle(output)
        return output





if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EMFFTransNet(classes=11).to(device)
    input = torch.zeros((1, 3, 360, 480)).to(device)
    flops, params = profile(model.to(device), inputs=(input,))
 
    print("params", params)
    print("FLOPS：", flops)


    
