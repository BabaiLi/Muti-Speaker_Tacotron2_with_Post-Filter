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

import numpy as np
import os
import random
from pathlib import Path

import torch
import torchaudio

from glob import glob
from torch.utils.data.distributed import DistributedSampler


class NumpyDataset(torch.utils.data.Dataset):
  def __init__(self, mel_paths, target_paths):
    super().__init__()
    self.mel    = []
    self.target = []
    self.mel_paths = Path(mel_paths[0])
    self.target_paths = Path(target_paths[0])
        
    for mel_path in self.mel_paths.iterdir():
        mel_list = [m for m in mel_path.iterdir()]
        random.shuffle(mel_list)
        for mel in mel_list:
            t_path = Path(self.target_paths, mel.parents[0].name, mel.name)
            if not t_path.is_file():
                t_path = Path(self.target_paths, mel.parents[0].name, mel.name+'.npy')
            self.mel += [mel]
            self.target += [t_path]
    

  def __len__(self):
    return len(self.mel)

  def __getitem__(self, idx):
    mel_filename  = self.mel[idx]
    target_filename = self.target[idx]
    
    if mel_filename.suffix == '.npy':
        spectrogram = torch.from_numpy(np.load(mel_filename))
    else:
        spectrogram = torch.load(mel_filename)
    if spectrogram.ndim == 3:
        spectrogram = spectrogram[0]
    spectrogram = torch.exp(spectrogram)
    spectrogram = torch.log10(spectrogram) * 20 - 20
    spectrogram = torch.clip((spectrogram + 100) / 100, min=0, max=1)
    
    if target_filename.suffix == '.npy':
        target = torch.from_numpy(np.load(target_filename))
    else:
        target = torch.load(target_filename)
    if target.ndim == 3:
        target = target[0]
    target = torch.exp(target)
    target = torch.log10(target) * 20 - 20
    target = torch.clip((target + 100) / 100, min=0, max=1)
    

    return {
        'target': target,
        'spectrogram': spectrogram
        }


class Collator:
  def __init__(self, params):
    self.params = params

  def collate(self, minibatch):
    samples_per_frame = self.params.hop_samples
    max_mel_len   = max([record['target'].size(1) for record in minibatch])
    mel_padded    = torch.ones(len(minibatch), 80, max_mel_len)
    target_padded = torch.ones(len(minibatch), 80, max_mel_len)
    
    for i in range(len(minibatch)):
        mel    = minibatch[i]['spectrogram']
        target = minibatch[i]['target']
        
        mel_padded[i, :, :mel.size(1)]       = mel
        target_padded[i, :, :target.size(1)] = target

    return {
        'target': target_padded,
        'spectrogram': mel_padded
        }


def from_path(mel_dirs, target_dirs, params, is_distributed=False):
  dataset = NumpyDataset(mel_dirs, target_dirs)
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate,
      shuffle=not is_distributed,
      num_workers=os.cpu_count() // 2,
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)
