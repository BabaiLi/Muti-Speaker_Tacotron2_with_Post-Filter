from dataclasses import dataclass

@dataclass
class Parameter:
    batch = 8
    seed  = 1234
    
    total_steps    = 100000
    save_per_iters = 1000
    
    use_saved_learning_rate = False
    ignore_layers           = ["embedding"]
    
    text_cleaners = ["transliteration_cleaners"]
    cmudict_path  = "data/cmu_dictionary"
    
    train   = "filelists/train.txt"
    val     = "filelists/val.txt"
    combine = "filelists/combine.txt"
    
    load_from_numpy   = True

@dataclass    
class Use_Speaker:
    use_spk_emb       = True
    use_spk_table     = False
    add_spk_to_prenet = True

    spk_emb = "128_aishell" # or None

@dataclass
class Optimizer:
    learning_rate    = 5e-5
    weight_decay     = 1e-7
    grad_clip_thresh = 1.0
    
@dataclass
class Model(Use_Speaker):
    # Init model
    mask_padding  = True
    mel_channels  = 80
    
    symbols_embedding = 512
    encoder_linear    = 64
    num_heads         = 2
    n_spk             = 2000
    spk_emb_in        = 128
    spk_emb_out       = 128
    
@dataclass
class Encoder:
    embedding       = 512
    n_convolutions  = 3
    kernel_size     = 5
    
@dataclass
class Decoder(Model):
    attention_rnn_dim = 1024
    decoder_rnn_dim   = 1024
    prenet_dim        = 256
    max_decoder_steps = 1000
    gate_threshold    = 0.5
    attention_dim     = 128
