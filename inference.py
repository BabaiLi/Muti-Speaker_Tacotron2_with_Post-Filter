from config import Parameter
from pypinyin import lazy_pinyin, Style
from pathlib import Path
import re
import numpy as np
import torch

from Tacotron2 import Tacotron2
from text import text_to_sequence, cmudict
from text.symbols import symbols
from Diff.inference import predict as diff_predict

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint_path',type=str, default='outdir/checkpoint_100000',   help='Type iterations')
parser.add_argument('-d', '--diff_model',     type=str, default=None, help='Type roman pinyin')
parser.add_argument('-s', '--speaker',        type=str, default=None,        help='choose speaker ID')
parser.add_argument('-e', '--use_extern_spk', type=str, default=None,          help='Type where wav_path')
args = parser.parse_args()

from HiFi_GAN.hifigan import Generator
from HiFi_GAN.config import Parameter as HiFi_Parameter
from scipy.io.wavfile import write
generator = Generator(HiFi_Parameter).to('cuda')
state_dict_g = torch.load('./HiFi_GAN/g_02500000', map_location='cuda')
generator.load_state_dict(state_dict_g['generator'])
generator.eval()
generator.remove_weight_norm()

checkpoint_path = args.checkpoint_path
model = Tacotron2(len(symbols))
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().float()

speaker = args.speaker
if args.use_extern_spk is None:
    if args.use_spk_emb:
        print('使用內部語者')
        if Parameter.spk_emb is not None:
            speaker = torch.load(Parameter.spk_emb)[speaker].cuda()
else:
    from tqdm import tqdm
    import torchaudio
    print('使用外部音檔')
    import sys
    sys.path.append('AdaIn-VC/')
    from data import Wav2Mel
    wav2mel = Wav2Mel()
    model_path = 'AdaIn-VC/model.ckpt'
    vc = torch.jit.load(model_path).to('cuda')
    if Path.is_dir(args.use_extern_spk):
        print('發現外部資料夾，開始抓取語者Embedding')
        f = Path.iterdir(args.extern)
        speaker = 0
        for i in tqdm(f):
            audio, sr = torchaudio.load(args.extern+'/'+i)
            src = wav2mel(audio, sr)[None, :]
            with torch.no_grad():
                _, emb = vc.inference(src.cuda(), src.cuda())
            speaker += emb
        speaker = speaker / len(f)
    else:
        audio, sr = torchaudio.load(args.extern)
        src = wav2mel(audio, sr)[None, :]
        with torch.no_grad():
            _, speaker = vc.inference(src.cuda(), src.cuda())
        

with open('sentence.txt', 'r') as f:
    txt = f.readlines()
name = 1

for i in txt:
    sentence = i[:-1]
    if not re.match(r'[a-z]+', sentence.lower(), re.I):
        number = ['1','2','3','4','5']
        t = ''
        for i in sentence:
            pinyin = lazy_pinyin(i, style=Style.TONE3)
            if pinyin[0][-1] in number:
                t = t + ' ' + pinyin[0]
            else:
                t = t + ' ' + pinyin[0]+'5'
        t = t.strip(" ")
    else:
        t = sentence
    t = t + '.'

    arpabet_dict = cmudict.CMUDict(Parameter.cmudict_path)
    text_cleaners = Parameter.text_cleaners
    sequence = torch.LongTensor(text_to_sequence(t, text_cleaners, arpabet_dict)).unsqueeze(0).cuda()
    
    method = ['_pn_']
    storge = []
    with torch.no_grad():
        mel_outputs, _, alignments, self_alignments = model.inference(sequence, speaker)
        storge.append(mel_outputs)
        mel = torch.exp(mel_outputs)
        mel = torch.log10(mel) * 20 - 20
        mel = (mel + 100) / 100
        if args.diff_model is not None:
            mel, sr = diff_predict(mel, model_dir=args.diff_model, fast_sampling=False)
            storge.append(mel)
            mel = torch.clamp(mel, -0.2, 0.95)
            mel = mel * 100 - 100
            mel = (mel + 20) / 20
            mel = 10**mel
            mel = torch.log(mel)
            method.append('_pf_')
        dir_name = args.speaker
        while len(dir_name) != 4:
            dir_name = '0' + dir_name
        for i in range(len(storge)):
            y_g_hat = generator(storge[i]).squeeze()
            audio = y_g_hat * 32767.0
            audio = audio.cpu().numpy().astype('int16')
            path = Path('inference', f"{dir_name}{method[i]}{t}wav")
            write(path, 22050, audio)
            name += 1
        print("\n", t)
