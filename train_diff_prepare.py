from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from config import Parameter
import argparse
import torch
import numpy as np

from Tacotron2 import Tacotron2
from text import symbols, text_to_sequence, cmudict

def parse_batch(batch: Tuple) -> Tuple:
    text_padded, input_lengths, mel_padded, gate_padded, \
    output_lengths, speaker, text_len, mel_len = batch

    text_padded = torch.autograd.Variable(text_padded.contiguous().cuda(non_blocking=True)).long()
    input_lengths = torch.autograd.Variable(input_lengths.contiguous().cuda(non_blocking=True)).long()
    max_len = torch.max(input_lengths.data).item()
    mel_padded = torch.autograd.Variable(mel_padded.contiguous().cuda(non_blocking=True)).float()
    gate_padded = torch.autograd.Variable(gate_padded.contiguous().cuda(non_blocking=True)).float()
    if speaker is not None:
        speaker = torch.autograd.Variable(speaker.contiguous().cuda(non_blocking=True)).float()
    output_lengths = torch.autograd.Variable(output_lengths.contiguous().cuda(non_blocking=True)).long()
    return ((text_padded, input_lengths, mel_padded, max_len, output_lengths, speaker),
            (mel_padded, gate_padded),
            (text_len, mel_len))
    

def main(checkpoint_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(Parameter.seed)
    torch.cuda.manual_seed(Parameter.seed)
    
    data_path = Path(Parameter.combine)
    with open(data_path, 'r') as f:
        txt = f.readlines()
    
    model = Tacotron2(len(symbols)).to(device)

    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
                
    _ = model.cuda().eval().float()
    if Parameter.use_spk_emb:
        if not Parameter.use_spk_table:
            spk_emb = torch.load(Parameter.spk_emb)
    speaker = None
    
    for data in tqdm(txt):
        audiopath, text, ids = data.split('|')
        audiopath = Path(audiopath)
        
        mel  = torch.from_numpy(np.load(audiopath))
        mel  = mel.unsqueeze(0).float()
        name = audiopath.name
        text = torch.IntTensor(text_to_sequence(text, Parameter.text_cleaners,
                                   cmudict.CMUDict(Parameter.cmudict_path)))
        text = text.unsqueeze(0).long()
        
        if Parameter.use_spk_emb:
            if not Parameter.use_spk_table:
                speaker = spk_emb[ids].float()
        
        
        input_length = torch.LongTensor([len(text[0])])
        gate  = torch.zeros(1, mel.size(2)).float()
        gate[0, :mel.size(2)-1] = 1
        output_length = torch.LongTensor([mel.size(2)])
        batch = (text, input_length, mel, gate, output_length, speaker, input_length, output_length)
        
        x, y, z = parse_batch(batch)
        y_pred = model(x)
        mel_output = y_pred[0]
        
        save_path = Path('diff_data', audiopath.stem[0:7])
        if not save_path.exists():
            Path.mkdir(save_path, parents=True)
        save_path = Path(save_path, audiopath.stem)
        
        torch.save(mel_output.cpu().detach(), save_path)
         

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    
    assert torch.cuda.is_available(), "must use Cuda! or the training won't start!"
    main(**vars(parser.parse_args()))
