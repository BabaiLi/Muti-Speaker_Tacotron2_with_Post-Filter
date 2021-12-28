from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torchaudio

from data import Wav2Mel

model_path = 'model.ckpt'
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load(model_path).to(device)
wav2mel = Wav2Mel()
    
path = Path('/media/kiwiiii/Data1/Corpus/aishell3/aishell3/wav')
embedding = {}

for i in tqdm(path.iterdir()):
    file_path = Path(path, i)
    file_path = file_path.iterdir()
    value=0
    for (count, j) in enumerate(file_path):
        src, src_sr = torchaudio.load(j)
        src = wav2mel(src, src_sr)[None, :].to(device)
        with torch.no_grad():
            cvt, emb = model.inference(src, src)
        value +=emb
    embedding[ str(int(i[3:])) ] = (v / (count+1)).cpu()
        
torch.save(embedding, 'embeddingg')
