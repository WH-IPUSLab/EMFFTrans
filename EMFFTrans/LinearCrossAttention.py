import torch
from torch import Tensor
from typing import Optional, Tuple
from torch.nn import functional as F
from torch.nn import Conv2d, Dropout
import torch.nn as nn
from EMFFTrans.BaseLayers import LayerNorm

class LinearCrossAttention(nn.Module):


    def __init__(
        self,
        channel_num,
        attention_dropout_rate: Optional[float] = 0.1,
        bias: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        channel_total_num= channel_num[0] + channel_num[1] +channel_num[2] +channel_num[3]
        self.channel_num_1 = channel_num[0]
        self.channel_num_2 = channel_num[1]
        self.channel_num_3 = channel_num[2]
        self.channel_num_4 = channel_num[3]
        
        self.q_proj=  Conv2d(
            in_channels=channel_total_num,
            out_channels=1,
            bias=bias,
            kernel_size=1,
            padding=0,
        )
        
        
        self.kv_1_proj = Conv2d(
            in_channels=channel_num[0],
            out_channels= (2 * channel_num[0]),
            bias=bias,
            kernel_size=1,
            padding=0,
        )
        self.kv_2_proj = Conv2d(
            in_channels=channel_num[1],
            out_channels= (2 * channel_num[1]),
            bias=bias,
            kernel_size=1,
            padding=0,
        )
        self.kv_3_proj = Conv2d(
            in_channels=channel_num[2],
            out_channels= (2 * channel_num[2]),
            bias=bias,
            kernel_size=1,
            padding=0,
        )
        self.kv_4_proj = Conv2d(
            in_channels=channel_num[3],
            out_channels= (2 * channel_num[3]),
            bias=bias,
            kernel_size=1,
            padding=0,
        )

        self.attn_dropout = Dropout(p=attention_dropout_rate)
        self.out_proj_1 = Conv2d(
            in_channels=channel_num[0],
            out_channels=channel_num[0],
            bias=bias,
            kernel_size=1,
            padding=0,
        )
        self.out_proj_2 = Conv2d(
            in_channels=channel_num[1],
            out_channels=channel_num[1],
            bias=bias,
            kernel_size=1,
            padding=0,
        )
        self.out_proj_3 = Conv2d(
            in_channels=channel_num[2],
            out_channels=channel_num[2],
            bias=bias,
            kernel_size=1,
            padding=0,
        )
        self.out_proj_4 = Conv2d(
            in_channels=channel_num[3],
            out_channels=channel_num[3],
            bias=bias,
            kernel_size=1,
            padding=0,
        )



    def forward(self,x1,x2,x3,x4,x_total):
        
        
        
        x1=x1.unsqueeze(-2).permute(0, 3, 2, 1) 
        x2=x2.unsqueeze(-2).permute(0, 3, 2, 1) 
        x3=x3.unsqueeze(-2).permute(0, 3, 2, 1) 
        x4=x4.unsqueeze(-2).permute(0, 3, 2, 1) 
        x_total=x_total.unsqueeze(-2).permute(0, 3, 2, 1) 
        batch_size, in_dim, kv_patch_area, kv_num_patches = x1.shape

        q_patch_area, q_num_patches = x1.shape[-2:]

        assert (
            kv_patch_area == q_patch_area
        ), "The number of pixels in a patch for query and key_value should be the same"

        # compute query, key, and value
        # [B, C, P, M] --> [B, 1 + d, P, M]
        query = self.q_proj(x_total)
        kv_1 = self.kv_1_proj(x1)
        kv_2 = self.kv_2_proj(x2)
        kv_3 = self.kv_3_proj(x3)
        kv_4 = self.kv_4_proj(x4)

        key_1,value_1 = torch.split(kv_1, split_size_or_sections=[self.channel_num_1, self.channel_num_1], dim=1)
        key_2,value_2 = torch.split(kv_2, split_size_or_sections=[self.channel_num_2, self.channel_num_2], dim=1)
        key_3,value_3 = torch.split(kv_3, split_size_or_sections=[self.channel_num_3, self.channel_num_3], dim=1)
        key_4,value_4 = torch.split(kv_4, split_size_or_sections=[self.channel_num_4, self.channel_num_4], dim=1)
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)
        
        context_vector_1 = key_1 * context_scores
        context_vector_2 = key_2 * context_scores
        context_vector_3 = key_3 * context_scores
        context_vector_4 = key_4 * context_scores
        
        
        context_vector_1 = torch.sum(context_vector_1, dim=-1, keepdim=True)
        context_vector_2 = torch.sum(context_vector_2, dim=-1, keepdim=True)
        context_vector_3 = torch.sum(context_vector_3, dim=-1, keepdim=True)
        context_vector_4 = torch.sum(context_vector_4, dim=-1, keepdim=True)
        
        out_1 = F.relu(value_1) * context_vector_1.expand_as(value_1)
        out_2 = F.relu(value_2) * context_vector_2.expand_as(value_2)
        out_3 = F.relu(value_3) * context_vector_3.expand_as(value_3)
        out_4 = F.relu(value_4) * context_vector_4.expand_as(value_4)
        
        
        out_1 = self.out_proj_1(out_1)
        out_2 = self.out_proj_2(out_2)
        out_3 = self.out_proj_3(out_3)
        out_4 = self.out_proj_4(out_4)
        
        out_1=out_1.permute(0, 3, 2, 1).squeeze(-2)
        out_2=out_2.permute(0, 3, 2, 1).squeeze(-2)
        out_3=out_3.permute(0, 3, 2, 1).squeeze(-2)
        out_4=out_4.permute(0, 3, 2 ,1).squeeze(-2)


        return out_1,out_2,out_3,out_4,[]


