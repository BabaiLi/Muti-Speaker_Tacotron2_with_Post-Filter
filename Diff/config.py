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
from dataclasses import dataclass

@dataclass
class Parameter:
    # Training params
    batch_size=8
    learning_rate=0.0001
    max_grad_norm=None

    # Data params
    sample_rate=22050
    n_mels=80
    n_fft=1024
    hop_samples=256
    crop_mel_frames=128 # doesn't use.
    
    # Model Params
    residual_layers=10      # In my study, higer of layers will get #better# audio quality.
    residual_channels=192   # 128 or lower dim won't convergence.
    dilation_cycle_length=1 # In my study, higer of length will get #lower## audio quality.
    noise_schedule=np.linspace(1e-4, 0.05, 200).tolist()
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5]
