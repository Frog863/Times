import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from Timesnet.layers.Embed import DataEmbedding
from Timesnet.layers.Conv_Blocks import Inception_Block_V1
from Timesnet.layers.transformer_encoder import TransformerEncoder
from IPython import embed
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # embed()
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean  # seasonal trend

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)#维度T变为T/2+1 根据rfft规定变换完是虚数对称，所以保留一半加1
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)#取模batch维度平均，通道维度平均
    frequency_list[0] = 0# 将第一个频率值设置为0(直流分量)
    _, top_list = torch.topk(frequency_list, k)# 在频率列表中找到前K个最大值的索引,索引代表，避免高频噪声影响
    # embed()
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list#采样点数除幅值对应位置 face(B,62,64)
    # embed()
    return period, abs(xf).mean(-1)[:, top_list]#周期长度和
# data = torch.rand(16,200,3)
# # a,b = FFT_for_Period(data, k=2)
# def get_input_size():
# embed()

class TimesBlock1(nn.Module):
    def __init__(self, configs):
        super(TimesBlock1, self).__init__()
        self.seq_len = configs.seq_len#先验序列长度
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.cutpos = configs.cutpos
        self.cutnom = configs.cutnom
        # parameter-efficient design
        self.conv_s = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )#两层Inception_Block_V1
        self.conv_t = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )  # 两层Inception_Block_V1
        self.encoder_season = TransformerEncoder(configs.input_size1, configs.hidden_size, configs.num_layers, configs.num_heads, configs.dropout)
        self.encoder_trend = TransformerEncoder(configs.input_size1, configs.hidden_size, configs.num_layers,configs.num_heads, configs.dropout)
    def forward(self, Season,Trend):
        B, T, N = Season.size()
        period_list, period_weight = FFT_for_Period(Season, self.k)#period_list周期和权重
        period_list = [self.cutpos+1,self.cutpos+20,self.cutpos+30]
        period_weight = torch.ones_like(period_weight)
        device = torch.device("cuda:0")
        period_weight = period_weight.to(device)
        Season_res = []
        Trend_res = []
        # for i in range(self.k):
        #     period = period_list[i]
        #     # padding
        #     if (self.seq_len + self.pred_len) % period != 0:#如果不能整除
        #         length = (
        #                          ((self.seq_len + self.pred_len) // period) + 1) * period#补全一个周期，使得能整除
        #         padding = torch.zeros([Season.shape[0], (length - (self.seq_len + self.pred_len)), Season.shape[2]]).to(Season.device)#(length - (self.seq_len + self.pred_len)是补了多少0
        #         Season_out = torch.cat([Season, padding], dim=1)
        #         Trend_out = torch.cat([Trend, padding], dim=1)
        #     else:#如果能整除不用补0
        #         length = (self.seq_len + self.pred_len)
        #         Season_out = Season
        #         Trend_out = Trend
        #     # reshape
        #     # embed()
        #
        #     Season_out = Season_out.reshape(B, length // period, period,
        #                       N).permute(0, 3, 1, 2).contiguous()#变为2维张量，对调整后的张量进行维度置换，将维度的顺序调整为 (B, N, length // period, period)，将张量变为连续内存的形式，以确保后续操作的正确性
        #     Trend_out = Trend_out.reshape(B, length // period, period,
        #                                     N).permute(0, 3, 1, 2).contiguous()
        #     # 2D conv: from 1d Variation to 2d Variation
        #     # embed()
        #     Season_out = self.conv_s(Season_out)
        #     Trend_out = self.conv_t(Trend_out)
        #     Season_out = Season_out.permute(0, 2, 3, 1).reshape(B, -1, N)#将维度顺序还原，2维变一维，-1是自动计算维度大小
        #     Trend_out = Trend_out.permute(0, 2, 3, 1).reshape(B, -1, N)
        #
        #     # reshape back
        #     Season_res.append(Season_out[:, :(self.seq_len + self.pred_len), :])#
        #     Trend_res.append(Trend_out[:, :(self.seq_len + self.pred_len), :])  #
        #
        # Season_res = torch.stack(Season_res, dim=-1)#在res后加维度，数值为res中张量个数
        # Trend_res = torch.stack(Trend_res, dim=-1)
        # # adaptive aggregation
        # period_weight = F.softmax(period_weight, dim=1)
        # period_weight = period_weight.unsqueeze(
        #     1).unsqueeze(1).repeat(1, T, N, 1)
        # # embed()
        # Season_res = torch.sum(Season_res * period_weight, -1)#残差
        # Trend_res = torch.sum(Trend_res * period_weight, -1)
        # # residual connection
        #
        # Season_out_res = Season_res + Season
        # Trend_out_res = Trend_res + Trend
        Season_out_res = Season
        Trend_out_res = Trend
        ######注意力token切分
        if (self.seq_len + self.pred_len) % period_list[self.cutnom] != 0:  # 如果不能整除
            length = (
                             ((self.seq_len + self.pred_len) // period_list[self.cutnom]) + 1) * period_list[self.cutnom]  # 补全一个周期，使得能整除
            padding = torch.zeros([Season_out_res.shape[0], (length - (self.seq_len + self.pred_len)), Season_out_res.shape[2]]).to(
                Season_out_res.device)  # (length - (self.seq_len + self.pred_len)是补了多少0
            Season_out = torch.cat([Season_out_res, padding], dim=1)
            Trend_out = torch.cat([Trend_out_res, padding], dim=1)
        else:  # 如果能整除不用补0
            length = (self.seq_len + self.pred_len)
            Season_out = Season_out_res
            Trend_out = Trend_out_res
        # embed()
        Season_out = Season_out.reshape(B, length // period_list[self.cutnom], period_list[self.cutnom]*N).permute(0,2,1).contiguous()
        Trend_out = Trend_out.reshape(B, length // period_list[self.cutnom], period_list[self.cutnom]*N).permute(0,2,1).contiguous()

        Season_out_att = self.encoder_season(Season_out,Trend_out,Season_out)
        Trend_out_att = self.encoder_season(Trend_out, Season_out,Trend_out)
        Season_out = Season_out_att.permute(0, 2, 1).reshape(B, -1, N)
        Trend_out = Trend_out_att.permute(0, 2, 1).reshape(B, -1, N)
        Season_out = Season_out[:, :(self.seq_len + self.pred_len), :]
        Trend_out = Trend_out[:, :(self.seq_len + self.pred_len), :]
        return Season_out , Trend_out
class TimesBlock2(nn.Module):
    def __init__(self, configs):
        super(TimesBlock2, self).__init__()
        self.seq_len = configs.seq_len#先验序列长度
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.cutpos = configs.cutpos
        self.cutnom = configs.cutnom
        # parameter-efficient design
        self.conv_s = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )#两层Inception_Block_V1
        self.conv_t = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )  # 两层Inception_Block_V1
        self.encoder_season = TransformerEncoder(configs.input_size2, configs.hidden_size, configs.num_layers, configs.num_heads, configs.dropout)
        self.encoder_trend = TransformerEncoder(configs.input_size2, configs.hidden_size, configs.num_layers,configs.num_heads, configs.dropout)
    def forward(self, Season,Trend):
        B, T, N = Season.size()
        period_list, period_weight_i = FFT_for_Period(Season, self.k)#period_list周期和权重
        period_list = [self.cutpos+1,self.cutpos+15,self.cutpos+25]
        period_weight = torch.ones_like(period_weight_i)
        device = torch.device("cuda:0")
        period_weight = period_weight.to(device)
        Season_res = []
        Trend_res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:#如果不能整除
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period#补全一个周期，使得能整除
                padding = torch.zeros([Season.shape[0], (length - (self.seq_len + self.pred_len)), Season.shape[2]]).to(Season.device)#(length - (self.seq_len + self.pred_len)是补了多少0
                Season_out = torch.cat([Season, padding], dim=1)
                Trend_out = torch.cat([Trend, padding], dim=1)
            else:#如果能整除不用补0
                length = (self.seq_len + self.pred_len)
                Season_out = Season
                Trend_out = Trend
            # reshape
            # embed()

            Season_out = Season_out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()#变为2维张量，对调整后的张量进行维度置换，将维度的顺序调整为 (B, N, length // period, period)，将张量变为连续内存的形式，以确保后续操作的正确性
            Trend_out = Trend_out.reshape(B, length // period, period,
                                            N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            # embed()
            Season_out = self.conv_s(Season_out)
            Trend_out = self.conv_t(Trend_out)
            Season_out = Season_out.permute(0, 2, 3, 1).reshape(B, -1, N)#将维度顺序还原，2维变一维，-1是自动计算维度大小
            Trend_out = Trend_out.permute(0, 2, 3, 1).reshape(B, -1, N)

            # reshape back
            Season_res.append(Season_out[:, :(self.seq_len + self.pred_len), :])#
            Trend_res.append(Trend_out[:, :(self.seq_len + self.pred_len), :])  #

        Season_res = torch.stack(Season_res, dim=-1)#在res后加维度，数值为res中张量个数
        Trend_res = torch.stack(Trend_res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        # embed()
        # print('period_weight',period_weight.shape)
        # print('res',res.shape)
        Season_res = torch.sum(Season_res * period_weight, -1)#残差
        Trend_res = torch.sum(Trend_res * period_weight, -1)
        # residual connection

        Season_out_res = Season_res + Season
        Trend_out_res = Trend_res + Trend
        ######注意力token切分
        if (self.seq_len + self.pred_len) % period_list[self.cutnom] != 0:  # 如果不能整除
            length = (
                             ((self.seq_len + self.pred_len) // period_list[self.cutnom]) + 1) * period_list[self.cutnom]  # 补全一个周期，使得能整除
            padding = torch.zeros([Season_out_res.shape[0], (length - (self.seq_len + self.pred_len)), Season_out_res.shape[2]]).to(
                Season_out_res.device)  # (length - (self.seq_len + self.pred_len)是补了多少0
            Season_out = torch.cat([Season_out_res, padding], dim=1)
            Trend_out = torch.cat([Trend_out_res, padding], dim=1)
        else:  # 如果能整除不用补0
            length = (self.seq_len + self.pred_len)
            Season_out = Season_out_res
            Trend_out = Trend_out_res
        # embed()
        Season_out = Season_out.reshape(B, length // period_list[self.cutnom], period_list[self.cutnom]*N).permute(0,2,1).contiguous()
        Trend_out = Trend_out.reshape(B, length // period_list[self.cutnom], period_list[self.cutnom]*N).permute(0,2,1).contiguous()
        Season_out_att = self.encoder_season(Season_out,Trend_out,Season_out)
        Trend_out_att = self.encoder_season(Trend_out, Season_out,Trend_out)
        Season_out = Season_out_att.permute(0, 2, 1).reshape(B, -1, N)
        Trend_out = Trend_out_att.permute(0, 2, 1).reshape(B, -1, N)
        Season_out = Season_out[:, :(self.seq_len + self.pred_len), :]
        Trend_out = Trend_out[:, :(self.seq_len + self.pred_len), :]
        return Season_out , Trend_out
class TimesBlock3(nn.Module):
    def __init__(self, configs):
        super(TimesBlock3, self).__init__()
        self.seq_len = configs.seq_len#先验序列长度
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.cutpos = configs.cutpos
        self.cutnom = configs.cutnom
        # parameter-efficient design
        self.conv_s = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )#两层Inception_Block_V1
        self.conv_t = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )  # 两层Inception_Block_V1
        self.encoder_season = TransformerEncoder(configs.input_size3, configs.hidden_size, configs.num_layers, configs.num_heads, configs.dropout)
        self.encoder_trend = TransformerEncoder(configs.input_size3, configs.hidden_size, configs.num_layers,configs.num_heads, configs.dropout)
    def forward(self, Season,Trend):
        B, T, N = Season.size()
        period_list, period_weight_i = FFT_for_Period(Season, self.k)#period_list周期和权重
        period_list = [self.cutpos+1,self.cutpos+5,self.cutpos+15]
        period_weight = torch.ones_like(period_weight_i)
        device = torch.device("cuda:0")
        period_weight = period_weight.to(device)
        Season_res = []
        Trend_res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:#如果不能整除
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period#补全一个周期，使得能整除
                padding = torch.zeros([Season.shape[0], (length - (self.seq_len + self.pred_len)), Season.shape[2]]).to(Season.device)#(length - (self.seq_len + self.pred_len)是补了多少0
                Season_out = torch.cat([Season, padding], dim=1)
                Trend_out = torch.cat([Trend, padding], dim=1)
            else:#如果能整除不用补0
                length = (self.seq_len + self.pred_len)
                Season_out = Season
                Trend_out = Trend
            # reshape
            # embed()

            Season_out = Season_out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()#变为2维张量，对调整后的张量进行维度置换，将维度的顺序调整为 (B, N, length // period, period)，将张量变为连续内存的形式，以确保后续操作的正确性
            Trend_out = Trend_out.reshape(B, length // period, period,
                                            N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            # embed()
            Season_out = self.conv_s(Season_out)
            Trend_out = self.conv_t(Trend_out)
            Season_out = Season_out.permute(0, 2, 3, 1).reshape(B, -1, N)#将维度顺序还原，2维变一维，-1是自动计算维度大小
            Trend_out = Trend_out.permute(0, 2, 3, 1).reshape(B, -1, N)

            # reshape back
            Season_res.append(Season_out[:, :(self.seq_len + self.pred_len), :])#
            Trend_res.append(Trend_out[:, :(self.seq_len + self.pred_len), :])  #

        Season_res = torch.stack(Season_res, dim=-1)#在res后加维度，数值为res中张量个数
        Trend_res = torch.stack(Trend_res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        # embed()
        # print('period_weight',period_weight.shape)
        # print('res',res.shape)
        Season_res = torch.sum(Season_res * period_weight, -1)#残差
        Trend_res = torch.sum(Trend_res * period_weight, -1)
        # residual connection

        Season_out_res = Season_res + Season
        Trend_out_res = Trend_res + Trend
        ######注意力token切分
        if (self.seq_len + self.pred_len) % period_list[self.cutnom] != 0:  # 如果不能整除
            length = (
                             ((self.seq_len + self.pred_len) // period_list[self.cutnom]) + 1) * period_list[self.cutnom]  # 补全一个周期，使得能整除
            padding = torch.zeros([Season_out_res.shape[0], (length - (self.seq_len + self.pred_len)), Season_out_res.shape[2]]).to(
                Season_out_res.device)  # (length - (self.seq_len + self.pred_len)是补了多少0
            Season_out = torch.cat([Season_out_res, padding], dim=1)
            Trend_out = torch.cat([Trend_out_res, padding], dim=1)
        else:  # 如果能整除不用补0
            length = (self.seq_len + self.pred_len)
            Season_out = Season_out_res
            Trend_out = Trend_out_res
        # embed()
        Season_out = Season_out.reshape(B, length // period_list[self.cutnom], period_list[self.cutnom]*N).permute(0,2,1).contiguous()
        Trend_out = Trend_out.reshape(B, length // period_list[self.cutnom], period_list[self.cutnom]*N).permute(0,2,1).contiguous()
        Season_out_att = self.encoder_season(Season_out,Trend_out,Season_out)
        Trend_out_att = self.encoder_season(Trend_out, Season_out,Trend_out)
        Season_out = Season_out_att.permute(0, 2, 1).reshape(B, -1, N)
        Trend_out = Trend_out_att.permute(0, 2, 1).reshape(B, -1, N)
        Season_out = Season_out[:, :(self.seq_len + self.pred_len), :]
        Trend_out = Trend_out[:, :(self.seq_len + self.pred_len), :]
        return Season_out , Trend_out


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.ronghe = nn.Linear(configs.data_len*2, configs.data_len)
        self.model = nn.ModuleList([TimesBlock1(configs),TimesBlock2(configs),TimesBlock3(configs)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers#TIMESBLOCK层数
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)#全连接，类别数
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C] 映射到固定通道数
        seasonal_init, trend_init = self.decompsition(enc_out)
        # TimesNet
        for i in range(self.layer):
            season_out, trend_out = self.model[i](seasonal_init, trend_init)
            season_out = self.layer_norm(season_out)
            trend_out = self.layer_norm(trend_out)
        output_season = self.act(season_out)
        output_trend = self.act(trend_out)
        ###############ronghe#######
        # output_season = output_season.permute(0, 2, 1)
        # output_trend = output_trend.permute(0, 2, 1)
        # output = torch.cat([output_season, output_trend], dim=2)
        # output = self.ronghe(output)
        # output = output.permute(0, 2, 1)
        ######################
        output = output_season + output_trend
        # output = output_season

        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)全部拉直
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None


