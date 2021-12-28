import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from text import text_to_sequence, cmudict
from config import Use_Speaker, Parameter

class DataSet(Dataset):
    def __init__(self, data_path: str):
        self.data_dir = data_path
        
        self.use_spk_emb   = Use_Speaker.use_spk_emb
        self.use_spk_table = Use_Speaker.use_spk_table
        
        self.text_cleaners = Parameter.text_cleaners
        self.cmudict = None
        if Parameter.cmudict_path is not None:
            self.cmudict = cmudict.CMUDict(Parameter.cmudict_path)
            
        self.load_from_numpy = Parameter.load_from_numpy
        if Use_Speaker.spk_emb is not None:
            self.spk_emb = torch.load(Use_Speaker.spk_emb)
        
        with open(data_path, encoding='utf-8') as f:
            self.meta_data = [line.strip() for line in f]
        
        random.seed(Parameter.seed)
        random.shuffle(self.meta_data)
    
    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners, self.cmudict))
        return text_norm
    
    def __len__(self):
        return len(self.meta_data)
        
    def __getitem__(self, index):
        check = len(self.meta_data[index].split('|'))
        if check == 3:
            audiopath, text, ids = self.meta_data[index].split('|')
            if self.use_spk_emb:
                ids = torch.IntTensor([int(ids)]) if self.use_spk_table else self.spk_emb[ids]
            else:
                ids = None
        elif check == 2:
            audiopath, text = self.meta_data[index].split('|')
            ids = None
        else:
            assert False, "Check your train.txt or val.txt format is correct!"
            
        if self.load_from_numpy:
            mel = torch.from_numpy(np.load(audiopath))
        else:
            mel = torch.load(audiopath)
            
        text = self.get_text(text)
        return (text, mel, ids)

class DataCollate():
    def __init__(self):
        self.n_frames_per_step = 1
          
    def __call__(self, data):
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in data]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]
        
        text_padded = torch.LongTensor(len(data), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = data[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
            
        num_mels = data[0][1].size(0)
        mel_len = [x[1].size(1) for x in data]
        max_target_len = max(mel_len)
        mel_len = torch.IntTensor(mel_len)
        text_len = [len(x[0]) for x in data]
        text_len = torch.IntTensor(text_len)
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0
        
        mel_padded = torch.zeros(len(data), num_mels, max_target_len)
        gate_padded = torch.zeros(len(data), max_target_len)
        output_lengths = torch.LongTensor(len(data))
        if data[0][2].size(-1) == 1:
            speaker = torch.zeros(len(data), 1)
        elif data[0][2].size(-1) == 128:
            speaker = torch.zeros(len(data), 128)
        
        for i in range(len(ids_sorted_decreasing)):
            mel = data[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            if data[i][2] is not None:
                speaker[i, :] = data[ids_sorted_decreasing[i]][2]
        
        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, speaker, text_len, mel_len
            
def infinite_iterator(dataloader):
    while True:
        for batch in iter(dataloader):
            yield batch
