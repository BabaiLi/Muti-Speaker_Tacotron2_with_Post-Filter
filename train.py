import os
import time
import argparse
import random
from pathlib import Path
from typing import Tuple
from tqdm.auto import trange

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Tacotron2 import Tacotron2
from alignment_loss import ForwardSumLoss
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from data_utils import DataSet, DataCollate, infinite_iterator
from text import symbols
from config import Parameter, Optimizer

def parse_batch(batch: Tuple) -> Tuple:
    text_padded, input_lengths, mel_padded, gate_padded, \
    output_lengths, speaker, text_len, mel_len = batch

    text_padded = torch.autograd.Variable(text_padded.contiguous().cuda(non_blocking=True)).long()
    input_lengths = torch.autograd.Variable(input_lengths.contiguous().cuda(non_blocking=True)).long()
    max_len = torch.max(input_lengths.data).item()
    mel_padded = torch.autograd.Variable(mel_padded.contiguous().cuda(non_blocking=True)).float()
    gate_padded = torch.autograd.Variable(gate_padded.contiguous().cuda(non_blocking=True)).float()
    if speaker is not None:
        speaker = torch.autograd.Variable(speaker.contiguous().cuda(non_blocking=True)).long()
    output_lengths = torch.autograd.Variable(output_lengths.contiguous().cuda(non_blocking=True)).long()
    return ((text_padded, input_lengths, mel_padded, max_len, output_lengths, speaker),
            (mel_padded, gate_padded),
            (text_len, mel_len))

def Tacotron2Loss(y_pred: Tuple, y: Tuple, z: Tuple, align_w: float) -> Tensor:
    text_len, mel_len = z[0], z[1]
    mel_out, gate_out, align, _s = y_pred
    
    align = align.unsqueeze(1)
    align_loss = ForwardSumLoss()(align, text_len, mel_len)
    align_loss = align_loss * align_w
    
    mel_target, gate_target = y[0], y[1]
    # Set requires_grad False, targets don't need gradient.
    mel_target.requires_grad = False
    gate_target.requires_grad = False
    gate_target = gate_target.view(-1, 1)
    
    gate_out = gate_out.view(-1, 1)
    mel_loss = nn.MSELoss()(mel_out, mel_target)
    gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
    
    return mel_loss + gate_loss + align_loss, mel_loss, gate_loss, align_loss
    

def main(output_dir: str, log_dir: str, checkpoint_path: str, warm_start: bool):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(Parameter.seed)
    torch.cuda.manual_seed(Parameter.seed)
    
    model = Tacotron2(len(symbols)).to(device)
    opt = Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=Optimizer.learning_rate,
                                 weight_decay=Optimizer.weight_decay)
    
    writer = SummaryWriter(Path(output_dir, log_dir))
    
    train_set = DataSet(Parameter.train)
    train_collate = DataCollate()
    train_loader = DataLoader(
                              train_set, num_workers=os.cpu_count() // 2,
                              batch_size=Parameter.batch, collate_fn=train_collate
                             )
    train_iter = infinite_iterator(train_loader)
    
    val_set = DataSet(Parameter.val)
    val_collate = DataCollate()
    val_loader = DataLoader(
                            val_set, num_workers=1,
                            batch_size=1, collate_fn=val_collate
                           )
    val_iter = infinite_iterator(val_loader)
    
    iteration = 0
    total = Parameter.total_steps + 1
    if checkpoint_path is not None:
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        if warm_start:            
            model_dict = checkpoint_dict['state_dict']
            if len(Parameter.ignore_layers) > 0:
                model_dict = {k: v for k, v in model_dict.items()
                              if k not in Parameter.ignore_layers}
                dummy_dict = model.state_dict()
                dummy_dict.update(model_dict)
                model_dict = dummy_dict
            model.load_state_dict(model_dict)
        else:
            model.load_state_dict(checkpoint_dict['state_dict'])
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
            _learning_rate_ = checkpoint_dict['learning_rate']
            iteration = checkpoint_dict['iteration']
            print("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, iteration))
            if Parameter.use_saved_learning_rate:
                learning_rate = _learning_rate_
            
            iteration += 1
            
    from HiFi_GAN.hifigan import Generator
    from HiFi_GAN.config import Parameter as hifi_parameter
    generator = Generator(hifi_parameter).to('cuda')
    state_dict_g = torch.load('./HiFi_GAN/g_02500000', map_location='cuda')
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    
    pbar = trange(iteration, total, initial=iteration, total=total, ncols=0)
    model.train()
    
    for step in pbar:
        align_w = 0.01 if step > 25000 else 0.0

        start = time.perf_counter()
        model.zero_grad()
        batch = next(train_iter)
        x, y, z = parse_batch(batch)
        
        y_pred = model(x)
         
        loss, mel_loss, gate_loss, align_loss = Tacotron2Loss(y_pred, y, z, align_w)
        reduced_loss = loss.item()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), Optimizer.grad_clip_thresh)
        optimizer.step()
        
        writer.add_scalar("Training loss", reduced_loss, step)
        writer.add_scalar("Grad.norm", grad_norm.item(), step)
        writer.add_scalar("Align loss", align_loss.item(), step)
        
        duration = time.perf_counter() - start
        pbar.set_postfix({"Mel loss": mel_loss.item(), "Align loss": align_loss.item(), "Gate loss": gate_loss.item(), "Duration": duration})
        
        if step % Parameter.save_per_iters == 0:
            model.eval()
            with torch.no_grad():
                batch_val = next(val_iter)
                x_val, y_val, _z = parse_batch(batch_val)
                y_pred_val = model.inference(x_val[0][0:1], x_val[-1][0])
            
                mel_outputs, _, alignments, self_alignments = y_pred_val
                mel_targets, gate_targets = y_val
                mel_train, alignment = y_pred[0], y_pred[2]
                mel_train = mel_train[0:1].double()
                mel_train = torch.where(mel_train != 0, mel_train, -11.5).float()
                mel_target = y[0]
                mel_target = mel_target[0:1].double()
                mel_target = torch.where(mel_target != 0, mel_target, -11.5).float()
                
            
                predit_audio = generator(mel_outputs).squeeze()
                predit_audio = torch.clamp(predit_audio, min=-1.0, max=1.0)
                target_audio = generator(mel_targets).squeeze()
                target_audio = torch.clamp(target_audio, min=-1.0, max=1.0)
                train_audio  = generator(mel_train[0:1]).squeeze()
                train_audio  = torch.clamp(train_audio, min=-1.0, max=1.0)
                train_target = generator(mel_target[0:1]).squeeze()
                train_target = torch.clamp(train_target, min=-1.0, max=1.0)
                
            model.train()
            
            idx = random.randint(0, alignments.size(0) - 1)
            writer.add_image(
                "Training_alignment",
                plot_alignment_to_numpy(alignment[idx].data.cpu().numpy().T),
                step, dataformats='HWC')
            writer.add_image(
                "Inference_alignment",
                plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
                step, dataformats='HWC')
            writer.add_image(
                "Inference_alignment_self",
                plot_alignment_to_numpy(self_alignments[idx].data.cpu().numpy().T),
                step, dataformats='HWC')
            writer.add_image(
                "Inference_mel_target",
                plot_spectrogram_to_numpy(mel_targets[0].data.cpu().numpy()),
                step, dataformats='HWC')
            writer.add_image(
                "Inference_mel_pred",
                plot_spectrogram_to_numpy(mel_outputs[0].data.cpu().numpy()),
                step, dataformats='HWC')
            writer.add_audio(
                "Train_target_audio",
                train_target, sample_rate=22050)
            writer.add_audio(
                "Train_predit_audio",
                train_audio, sample_rate=22050)
            writer.add_audio(
                "Inference_predit_audio",
                predit_audio, sample_rate=22050)
            writer.add_audio(
                "Inference_target_audio",
                target_audio, sample_rate=22050)
            
            checkpoint_path = Path(output_dir, f"checkpoint_{step}")
            link_name       = Path(output_dir, "weights.pt")
            torch.save({'iteration': step,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'learning_rate': Optimizer.learning_rate},
                         checkpoint_path)
            if Path.is_symlink(link_name):
                Path.unlink(link_name)
            link_name.symlink_to(f"checkpoint_{step}")
                
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='outdir')
    parser.add_argument("--log_dir", type=str, default='logdir')
    parser.add_argument("--checkpoint_path", required=False, type=str)
    parser.add_argument("--warm_start", action='store_true')
    
    assert torch.cuda.is_available(), "must use Cuda! or the training won't start!"
    main(**vars(parser.parse_args()))
