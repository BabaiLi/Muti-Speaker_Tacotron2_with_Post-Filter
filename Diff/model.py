# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from config import Parameter
from typing import Dict, List


class ConvNorm(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True, w_init_gain="linear"):
    super(ConvNorm, self).__init__()

    if padding is None:
      assert kernel_size % 2 == 1
      padding = int(dilation * (kernel_size - 1) / 2)

    self.conv = nn.Conv1d(
      in_channels,
      out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      dilation=dilation,
      bias=bias,
    )
    nn.init.kaiming_normal_(self.conv.weight)

  def forward(self, signal):
    conv_signal = self.conv(signal)
    return conv_signal

class LinearNorm(nn.Module):
  def __init__(self, in_features, out_features, bias=False):
    super(LinearNorm, self).__init__()
    self.linear = nn.Linear(in_features, out_features, bias)

    nn.init.xavier_uniform_(self.linear.weight)
    if bias:
      nn.init.constant_(self.linear.bias, 0.0)
   
  def forward(self, x):
    x = self.linear(x)
    return x

class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps: int):
    super().__init__()
    self.max_steps = max_steps

  def forward(self, diffusion_step: Tensor) -> Tensor:
    device = diffusion_step.device
    half_dim = self.max_steps // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = diffusion_step[:, None] * emb[None, :]
    x = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return x

class ResidualBlock(nn.Module):
  def __init__(self, n_mels: int, residual_channels: int, dilation: int):
    super().__init__()
    self.dilated_conv = ConvNorm(residual_channels, 2 * residual_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
    
    self.diffusion_projection = LinearNorm(residual_channels, residual_channels)
    
    self.conditioner_projection = ConvNorm(80, 2 * residual_channels, kernel_size=1)
    self.output_projection = ConvNorm(residual_channels, 2 * residual_channels, kernel_size=1)

  def forward(
              self, 
              x: Tensor, 
              spectrogram: Tensor, 
              diffusion_step: Tensor) -> List[Tensor]:
    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)

    # Global Conditioner
    conditioner = self.conditioner_projection(spectrogram)

    y = x + diffusion_step
    y = self.dilated_conv(y)
    y = y + conditioner

    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter) 

    y = self.output_projection(y)
    residual, skip = torch.chunk(y, 2, dim=1)
    return (x + residual) / math.sqrt(2.0), skip


class DiffWave(nn.Module):
  def __init__(self, params: Dict = Parameter):
    super().__init__()
    self.params = params
    self.input_projection = ConvNorm(80, params.residual_channels, kernel_size=1)
    
    self.diffusion_embedding = DiffusionEmbedding(params.residual_channels)
    self.mlp = nn.Sequential(
      LinearNorm(params.residual_channels, params.residual_channels * 4),
      nn.Mish(),
      LinearNorm(params.residual_channels * 4, params.residual_channels)
    )
    
    self.residual_layers = nn.ModuleList([
        ResidualBlock(params.n_mels, params.residual_channels, 2**(i % params.dilation_cycle_length))
        for i in range(params.residual_layers)
    ])
    self.skip_projection = ConvNorm(params.residual_channels, params.residual_channels, kernel_size=1)
    self.output_projection = ConvNorm(params.residual_channels, 80, kernel_size=1)
    nn.init.zeros_(self.output_projection.conv.weight)

  def forward(
              self, 
              noise: Tensor, 
              spectrogram: Tensor, 
              diffusion_step: Tensor) -> Tensor:
              
    x = self.input_projection(noise)
    x = F.relu(x)
    
    diffusion_step = self.diffusion_embedding(diffusion_step)
    diffusion_step = self.mlp(diffusion_step)
    
    skip = []
    for layer in self.residual_layers:
      x, skip_connection = layer(x, spectrogram, diffusion_step)
      skip.append(skip_connection)

    x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
    x = self.skip_projection(x)
    x = F.relu(x)
    x = self.output_projection(x)
    return x
