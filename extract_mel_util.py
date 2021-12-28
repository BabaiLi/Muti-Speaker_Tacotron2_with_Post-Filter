import torch
from torch_stft import STFT
import librosa
from librosa.filters import mel as librosa_mel_fn

import argparse
import soundfile as sf
import numpy as np

from tqdm import tqdm
from pathlib import Path

def extract_mel(s: str, output_dir: str, stft: torch.nn.Module, n_mel_channels: int, f_min: float, f_max: float):
    global device, save_to_np
    
    audio, sr = sf.read(s)
    audio     = torch.FloatTensor(audio).to(device)
    audio     = audio.unsqueeze(0)

    magnitude, phase = stft.transform(audio)
    mel_basis        = librosa_mel_fn(sr, stft.filter_length, n_mel_channels, f_min, f_max)
    mel_basis        = torch.from_numpy(mel_basis).float().to(device)
    mel_output       = torch.matmul(mel_basis, magnitude)
    mel_output       = torch.log(torch.clamp(mel_output, min=1e-5) * 1)
    melspec          = torch.squeeze(mel_output, 0).cpu()
    
    output = Path(output_dir, s.parent.name, s.stem)
    if not output.parent.exists():
        Path.mkdir(output.parent)
    
    if save_to_np:
        np.save(output, melspec.numpy())
    else:
        torch.save(melspec, output)

def main(inputs: str, output_dir: str, stft: torch.nn.Module, n_mel_channels: int, f_min: float, f_max: float):
    global load_from_numpy     
    
    inputs_path = Path(inputs)
    output_dir  = Path(output_dir)
    
    if not output_dir.exists():
        Path.mkdir(output_dir)
    
    if inputs_path.is_dir():
        inputs_path = inputs_path.iterdir()
    elif inputs_path.is_file():
        inputs_path = [inputs_path.name]
    else:
        print("Check your input, must be a file or dir!")
        
    for i in tqdm(inputs_path):
        if i.is_dir():
            for j in i.iterdir():
                if j.suffix == '.wav':
                    extract_mel(j, output_dir, stft, n_mel_channels, f_min, f_max)
                else:
                    print("Need input wav file!")
        else:
            if i.suffix == '.wav':
                extract_mel(i, output_dir, stft, n_mel_channels, f_min, f_max)
            else:
                print("Need input wav file!")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=str)
    parser.add_argument("--output_dir",     type=str, default="extract_mel")
    parser.add_argument("--filter_length",  type=int, default=1024)
    parser.add_argument("--win_length",     type=int, default=1024)
    parser.add_argument("--hop_length",     type=int, default=256)
    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--f_min",          type=float, default=20.0)
    parser.add_argument("--f_max",          type=float, default=8000.0)
    parser.add_argument("--window",         type=str, default='hann')
    parser.add_argument("--save_to_np",   action='store_true')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_to_np = args.save_to_np
    
    stft = STFT(
    filter_length = args.filter_length, 
    hop_length    = args.hop_length, 
    win_length    = args.win_length,
    window        = args.window
    ).to(device)
    
    main(args.inputs, args.output_dir, stft, args.n_mel_channels, args.f_min, args.f_max)
