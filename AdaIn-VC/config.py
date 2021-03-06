from dataclasses import dataclass

@dataclass
class SpeakerEncoder:
    c_in=80
    c_h=128
    c_out=128
    kernel_size=5
    bank_size=8
    bank_scale=1
    c_bank=128
    n_conv_blocks=6
    n_dense_blocks=6
    subsample=[1, 2, 1, 2, 1, 2]
    act="relu"
    dropout_rate=0.0

@dataclass
class ContentEncoder:
    c_in=80
    c_h=128
    c_out=128
    kernel_size=5
    bank_size=8
    bank_scale=1
    c_bank=128
    n_conv_blocks=6
    subsample=[1, 2, 1, 2, 1, 2]
    act="relu"
    dropout_rate=0.0

@dataclass
class Decoder:
    c_in=128
    c_cond=128
    c_h=128
    c_out=80
    kernel_size=5
    n_conv_blocks=6
    upsample=[2, 1, 2, 1, 2, 1]
    act="relu"
    sn=false
    dropout_rate=0.0

@dataclass
class Optimizer:
    lr=0.0005
    beta1=0.9
    beta2=0.999
    amsgrad=true
    weight_decay=0.0001
    grad_norm=5
    
@dataclass
class Lambda:
    rec=10
    kl=1
    kl_annealing=20000
