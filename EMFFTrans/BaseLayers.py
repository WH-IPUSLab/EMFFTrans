from typing import Union, Optional, List, Tuple
from EMFFTrans.model_utils import make_divisible

import torch
import torch.nn as nn
from torch import Tensor, Size
import torch.nn.functional as F

class GlobalPool(nn.Module):


    pool_types = ["mean", "rms", "abs"]

    def __init__(
        self,
        pool_type: Optional[str] = "mean",
        keep_dim: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.pool_type = pool_type
        self.keep_dim = keep_dim


    def _global_pool(self, x: Tensor, dims: List):
        if self.pool_type == "rms":  # root mean square
            x = x**2
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
            x = x**-0.5
        elif self.pool_type == "abs":  # absolute
            x = torch.mean(torch.abs(x), dim=dims, keepdim=self.keep_dim)
        else:
            # default is mean
            # same as AdaptiveAvgPool
            x = torch.mean(x, dim=dims, keepdim=self.keep_dim)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            dims = [-2, -1]
        elif x.dim() == 5:
            dims = [-3, -2, -1]
        else:
            raise NotImplementedError("Currently 2D and 3D global pooling supported")
        return self._global_pool(x, dims=dims)


class LayerNorm(nn.LayerNorm):


    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def forward(self, x: Tensor) -> Tensor:
        n_dim = x.ndim
        if x.shape[1] == self.normalized_shape[0] and n_dim > 2:  # channel-first format
            s, u = torch.std_mean(x, dim=1, keepdim=True, unbiased=False)
            x = (x - u) / (s + self.eps)
            if self.weight is not None:
                # Using fused operation for performing affine transformation: x = (x * weight) + bias
                n_dim = x.ndim - 2
                new_shape = [1, self.normalized_shape[0]] + [1] * n_dim
                x = torch.addcmul(
                    input=self.bias.reshape(*[new_shape]),
                    value=1.0,
                    tensor1=x,
                    tensor2=self.weight.reshape(*[new_shape]),
                )
            return x
        elif x.shape[-1] == self.normalized_shape[0]:  # channel-last format
            return super().forward(x)
        else:
            raise NotImplementedError(
                "LayerNorm is supported for channel-first and channel-last format only"
            )


class InvertedResidual(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        dilation: int = 1,
        skip_connection: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    padding=0,
                ),

            )

            block.add_module(
                name="exp_1x1_bn",
                module=nn.BatchNorm2d(num_features=hidden_dim),
            )

            block.add_module(
                name="exp_1x1_act",
                module=nn.LeakyReLU(negative_slope=0.1),
            )


        block.add_module(
            name="conv_3x3",
            module=nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim,
                dilation=dilation,
                padding=1,
            ),
        )
        block.add_module(
            name="conv_3x3_bn",
            module=nn.BatchNorm2d(num_features=hidden_dim),
        )
        block.add_module(
            name="conv_3x3_act",
            module=nn.LeakyReLU(negative_slope=0.1),
        )


        block.add_module(
            name="red_1x1",
            module=nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            ),
        )
        block.add_module(
            name="red_1x1_bn",
            module=nn.BatchNorm2d(num_features=out_channels),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)