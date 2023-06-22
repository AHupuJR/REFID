# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
'''
HINet: Half Instance Normalization Network for Image Restoration

@inproceedings{chen2021hinet,
  title={HINet: Half Instance Normalization Network for Image Restoration},
  author={Liangyu Chen and Xin Lu and Jie Zhang and Xiaojie Chu and Chengpeng Chen},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2021}
}
'''

import torch
import torch.nn as nn
import math
from basicsr.models.archs.arch_util import FAC_bias
from torch.nn import functional as F

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

class SAM_Event(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM_Event, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3+7, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img, voxel):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        # add event
        x2 = torch.sigmoid(self.conv3(torch.cat([img, voxel], dim=1)))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

class SingleMultiConnectEVHINet(nn.Module):

    def __init__(self, in_chn=3, ev_chn=6, wf=64, depth=3, fac_place=2, fac_kernel_size=1, fac_before_downsample=True, event_feature_transfer=False, relu_slope=0.2, hin_position_left=0, hin_position_right=4):
        super(SingleMultiConnectEVHINet, self).__init__()
        self.depth = depth
        self.fac_place = fac_place
        self.fac_before_downsample = fac_before_downsample
        self.event_feature_transfer = event_feature_transfer
        self.fac_kernel_size = fac_kernel_size
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
         # ev
        self.down_path_ev = nn.ModuleList()
        self.conv_ev1 = nn.Conv2d(ev_chn, wf, 3, 1, 1)

        prev_channels = self.get_input_chn(wf)
        for i in range(depth): #0,1,2,3,4
            use_HIN = True if hin_position_left <= i and i <= hin_position_right else False
            downsample = True if (i+1) < depth else False # i max = 3
            event_transfer_flag = True if i==0 and self.event_feature_transfer else False # add event_transfer to convs in first block
            self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_HIN=use_HIN))
            self.down_path_2.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_csff=downsample, use_HIN=use_HIN))
            # ev encoder
            if i < self.fac_place+1:
                self.down_path_ev.append(UNetEVConvBlock(prev_channels, (2**i) * wf, downsample , relu_slope, use_HIN=use_HIN)) # downsample=False if fac_before_downsample=True
                    # self.conv_ev_before_fac = nn.Conv2d((2**i) * wf, (2**i) * wf *(fac_kernel_size**2), 1, 1, 0) # !!

            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)): #3,2,1,0
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.skip_conv_1.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            prev_channels = (2**i)*wf
        self.sam12 = SAM(prev_channels)
        # self.sam12 = SAM_Event(prev_channels)

        self.cat12 = nn.Conv2d(prev_channels*2, prev_channels, 1, 1, 0)
        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def forward(self, x, event, mask=None):
        image = x

        ev = []
        #EVencoder
        e1 = self.conv_ev1(event)
        for i, down in enumerate(self.down_path_ev):
            # print('down.event_feature_transfer:{}'.format(down.event_feature_transfer))
            if i != self.fac_place:
                e1, e1_up = down(e1)
                if self.fac_before_downsample:
                    ev.append(e1_up)
                else:
                    ev.append(e1)
            else: # at loss 
                e1 = down(e1)
                ev.append(e1)



        # print('e1.shape{}'.format(e1.shape))

        #stage 1
        x1 = self.conv_01(image)
        encs = []
        decs = []
        masks = []
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:
                if i <= self.fac_place: # merge fac with backbone
                    x1, x1_up = down(x1, event_filter=ev[i], fac_kernel_size=self.fac_kernel_size, merge_before_downsample=self.fac_before_downsample)
                else:
                    x1, x1_up = down(x1)

                encs.append(x1_up)

                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor = 0.5**i))
            
                    
            else:
                x1 = down(x1)


        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1]))
            decs.append(x1)

        # sam_feature, out_1 = self.sam12(x1, image, event)
        sam_feature, out_1 = self.sam12(x1, image)


        # single version

        return [out_1]




    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


# origin
# class UNetConvBlock(nn.Module):
#     def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False, event_feature_transfer=False):
#         super(UNetConvBlock, self).__init__()
#         self.downsample = downsample
#         self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
#         self.use_csff = use_csff
#         self.event_feature_transfer = event_feature_transfer

#         self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
#         self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
#         self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
#         self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

#         if downsample and use_csff:
#             self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
#             self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)

#         if use_HIN:
#             self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
#         self.use_HIN = use_HIN

#         if downsample:
#             self.downsample = conv_down(out_size, out_size, bias=False)

#     def forward(self, x, enc=None, dec=None, event_filter=None, fac_kernel_size=None, merge_before_downsample=True):
#         out = self.conv_1(x)

#         if self.use_HIN:
#             out_1, out_2 = torch.chunk(out, 2, dim=1)
#             out = torch.cat([self.norm(out_1), out_2], dim=1)
#         out_conv1 = self.relu_1(out)
#         out_conv2 = self.relu_2(self.conv_2(out_conv1))

#         out = out_conv2 + self.identity(x)
#         if enc is not None and dec is not None:
#             assert self.use_csff
#             out = out + self.csff_enc(enc) + self.csff_dec(dec)
            
#         if event_filter is not None and fac_kernel_size is not None and merge_before_downsample:
#             out = FAC(out, event_filter, fac_kernel_size)
             
#         if self.downsample:
#             if self.event_feature_transfer:
#                 out_down = self.downsample(out)
#                 return out_down, out, out_conv1, out_conv2
#             else:
#                 out_down = self.downsample(out)
#                 return out_down, out

#         else:
#             if self.event_feature_transfer:
#                 return out, out_conv1, out_conv2
#             else:
#                 return out

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False, cat=False): # cat
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)
        
        if cat:
            self.conv_concat = nn.Conv2d(2*out_size, out_size, 1,1,0)

    def forward(self, x, enc=None, dec=None, mask=None, event_filter=None, fac_kernel_size=None, merge_before_downsample=True):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)

        # propose 1
        if enc is not None and dec is not None and mask is not None:
            assert self.use_csff
            out_enc = self.csff_enc(enc) + self.csff_enc_mask((1-mask)*enc)
            out_dec = self.csff_dec(dec) + self.csff_dec_mask(mask*dec)
            out = out + out_enc + out_dec
        
        # propose 2
        # if enc is not None and dec is not None and mask is not None:
        #     assert self.use_csff
        #     out_enc = (1-mask)*self.csff_enc(enc)
        #     out_dec = (mask)*self.csff_dec(dec)
        #     out = out + out_enc + out_dec
        
        # origin
        # if enc is not None and dec is not None:
            # assert self.use_csff
        #     out = out + self.csff_enc(enc) + self.csff_dec(dec)
            
        if event_filter is not None and fac_kernel_size is not None and merge_before_downsample:

            # out = FAC(out, event_filter, fac_kernel_size) # fac

            # out = out + event_filter # add

            # out = torch.cat((out, event_filter), dim=1) # concate
            # out = self.conv_concat(out)

            out = FAC_bias(out, event_filter)  # fac_bias
             
        if self.downsample:

            out_down = self.downsample(out)
            
            if not merge_before_downsample: # merge after downsample

                # out_down = FAC(out_down, event_filter, fac_kernel_size)# fac
                
                # out_down = out_down + event_filter # add

                # out_down = torch.cat((out_down, event_filter), dim=1) # concate
                # out_down = self.conv_concat(out_down)

                out_down = FAC_bias(out_down, event_filter) # fac_bias

            return out_down, out

        else:

            return out



class UNetEVConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetEVConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        # self.conv_before_merge = nn.Conv2d(out_size, out_size , 1, 1, 0)  # fac add concate
        self.conv_before_merge = nn.Conv2d(out_size, 2 * out_size , 1, 1, 0)  # fac_bias

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, fac_kernel_size=None, merge_before_downsample=True):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)
        
            
             
        if self.downsample:

            out_down = self.downsample(out)
            
            if not merge_before_downsample: # merge after downsample
            
                out_down = self.conv_before_merge(out_down)
            else : # merge before downsample
                out = self.conv_before_merge(out)
            return out_down, out

        else:

            out = self.conv_before_merge(out)
            return out



class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

class Subspace(nn.Module):

    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(UNetConvBlock(in_size, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x + sc


class skip_blocks(nn.Module):

    def __init__(self, in_size, out_size, repeat_num=1):
        super(skip_blocks, self).__init__()
        self.blocks = nn.ModuleList()
        self.re_num = repeat_num
        mid_c = 128
        self.blocks.append(UNetConvBlock(in_size, mid_c, False, 0.2))
        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock(mid_c, mid_c, False, 0.2))
        self.blocks.append(UNetConvBlock(mid_c, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for m in self.blocks:
            x = m(x)
        return x + sc


if __name__ == "__main__":
    pass
