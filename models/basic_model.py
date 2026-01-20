import torch
import torch.nn as nn
import warnings
import os
import math
import torch.nn.functional as F
from einops import rearrange

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# HCCM - CHP
class color_extra(nn.Module):
    def __init__(self, channels):
        super(color_extra, self).__init__()

        # conv, pooling, softmax
        self.block1 = nn.Sequential(nn.Conv2d(channels, 256, kernel_size=1, stride=1),
                                    nn.AdaptiveAvgPool2d((channels, 1)),
                                    nn.Softmax(dim=2))
        # linear
        self.linear = nn.Linear(256, channels)

        self.conv_out = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, raw, gt=None):

        b, c, h, w = raw.shape

        # in : raw -> out : raw hist
        hist = self.relu(self.linear(self.block1(raw).permute(0, 3, 2, 1).squeeze(1)))
        
        # in : gt -> out : gt hist
        if gt is not None:
            # revised : raw -> gt
            gt_hist = self.relu(self.linear(self.block1(gt).permute(0, 3, 2, 1).squeeze(1)))
            # debugging
            # if torch.abs(hist - gt_hist).sum() == 0:
                #print("Warning: pred_hist and gt_hist are EXACTLY the same!")
            #else:
                #print(f"Hist Diff Mean: {torch.abs(hist - gt_hist).mean().item():.10f}")
        else:
            gt_hist =None

        raw_reshape = rearrange(raw, 'b c h w -> b c (h w)')

        # out : Position-specific color feature F_c
        color_fea = hist @ raw_reshape
        color_fea = rearrange(color_fea, 'b c (h w) -> b c h w', h=h, w=w)
        color_fea_out = self.relu(self.conv_out(color_fea))

        return color_fea_out, hist, gt_hist

# HCCM - Attention
class Color_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Color_Attention, self).__init__()
        self.num_heads = num_heads
        self.q_layer = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias)

        self.k_layer = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias)

        self.v_layer = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), padding=0, bias=bias)

    def forward(self, input, color):
        b, c, h, w = input.shape

        q, k, v = self.q_layer(color), self.k_layer(input), self.v_layer(input)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        # out : F_s prediction before resblock
        return out


class HCCM(nn.Module):
    def __init__(self, channels):
        super(HCCM, self).__init__()

        self.blocks0 = Res_block(channels, channels)

        self.color_extra = color_extra(channels)

        self.color_attention = Color_Attention(channels, num_heads=8, bias=False)

        self.blocks1 = Res_block(channels, channels)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, gray, raw, gt=None):
        out = self.blocks0(gray) # in : gray prediction -> resblock
        color_fea, pred_hist, gt_hist = self.color_extra(raw, gt) # F_c, raw, gt hist
        out_colored = self.color_attention(out, color_fea) # q,k,v
        out_colored = self.blocks1(out_colored) # out : F_s prediction after resblock

        return out_colored, pred_hist, gt_hist


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_block, self).__init__()

        sequence = []

        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        ]

        self.model = nn.Sequential(*sequence)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        out = self.model(x) + self.conv(x)

        return out


class upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsampling, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out


class channel_down(nn.Module):
    def __init__(self, channels, down_channels=3):
        super(channel_down, self).__init__()

        self.conv0 = nn.Sequential(Res_block(channels * 4, channels * 2),
                                   Res_block(channels * 2, channels))

        self.conv1 = nn.Conv2d(channels, down_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.sigmoid(self.conv1(self.conv0(x)))

        return out


class channel_up(nn.Module):
    def __init__(self, channels, down_channels=3):
        super(channel_up, self).__init__()

        self.conv_0 = nn.Conv2d(down_channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.conv1 = nn.Sequential(Res_block(channels, channels * 2),
                                   Res_block(channels * 2, channels * 4))

        self.relu = nn.LeakyReLU()

    def forward(self, x):

        out = self.relu(self.conv1(self.relu(self.conv_0(x))))

        return out



class feature_pyramid(nn.Module):
    def __init__(self, channels):
        super(feature_pyramid, self).__init__()

        self.conv_in_ch1 = nn.Conv2d(1, channels, kernel_size=(5, 5), stride=(1, 1), padding=2)

        self.conv_in_ch3 = nn.Conv2d(3, channels, kernel_size=(5, 5), stride=(1, 1), padding=2)

        self.block0 = nn.Sequential(Res_block(channels, channels),
                                    Res_block(channels, channels * 2))

        self.down0 = nn.Conv2d(channels * 2, channels * 2, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block1 = nn.Sequential(Res_block(channels * 2, channels * 2),
                                    Res_block(channels * 2, channels * 4))

        self.down1 = nn.Conv2d(channels * 4, channels * 4, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):

        if x.shape[1] == 1:
            x = self.conv_in_ch1(x)
        else:
            x = self.conv_in_ch3(x)
        level0 = self.down0(self.block0(x))
        level1 = self.down1(self.block1(level0))

        return level0, level1



class AE(nn.Module):
    def __init__(self, channels=64, down_channels=3):
        super(AE, self).__init__()

        self.pyramid = feature_pyramid(channels)
        self.channel_down = channel_down(channels, down_channels = down_channels)
    
        self.channel_up = channel_up(channels, down_channels=down_channels)

        self.color = HCCM(channels * 4)

        #decoder

        self.block_up0 = Res_block(channels * 4, channels * 4)
        self.block_up1 = Res_block(channels * 4, channels * 4)
        self.up_sampling0 = upsampling(channels * 4, channels * 2)
        self.block_up2 = Res_block(channels * 2, channels * 2)
        self.block_up3 = Res_block(channels * 2, channels * 2)
        self.up_sampling1 = upsampling(channels * 2, channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.conv_out_rgb = nn.Conv2d(channels, 3, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv_out_raw = nn.Conv2d(channels, 1, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,y_gray=None, y=None, pred_y_gray=None):

        output = {}
        # =================encoder=================
        raw_fea2, raw_fea4 = self.pyramid(x)
        raw_fea_down4 = self.channel_down(raw_fea4)

        if pred_y_gray is None:
            gt_gray_fea_down4=None
            if y_gray is not None:
                gt_gray_fea2, gt_gray_fea4 = self.pyramid(y_gray)
                gt_gray_fea_down4 = self.channel_down(gt_gray_fea4)
            output['gt_gray_fea_down'] = gt_gray_fea_down4
            output['raw_fea_down'] = raw_fea_down4


        else:
            raw_fea_ori = self.channel_up(raw_fea_down4)
            pred_y_gray_ori = self.channel_up(pred_y_gray)

            gt_fea_ori = None
            if y is not None:
                gt_fea2, gt_fea4 = self.pyramid(y)
                gt_fea_ori = self.channel_up(self.channel_down(gt_fea4))
            recon_fea4, pred_hist, gt_hist = self.color(pred_y_gray_ori, raw_fea_ori, gt_fea_ori)



            recon_img_ori_up2 = self.up_sampling0(
                self.block_up1(self.block_up0(recon_fea4) + raw_fea4))
            recon_img_ori_up4 = self.up_sampling1(
                self.block_up3(self.block_up2(recon_img_ori_up2) + raw_fea2))

            recon_gt_img =self.sigmoid(self.conv_out_rgb(self.relu(self.conv2(recon_img_ori_up4))))
            output['recon_gt_img'] = recon_gt_img
            output['recon_gt_fea_ori'] = recon_fea4
            output['gt_fea_ori'] = gt_fea_ori
            output['pred_hist'] = pred_hist
            output['gt_hist']=gt_hist

        return output
