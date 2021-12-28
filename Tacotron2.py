from math import sqrt
from config import Model, Encoder, Decoder, Use_Speaker
from typing import Dict, Tuple, List, Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

drop_rate=0.5

# utils.py
###########################################################################
def get_mask_from_lengths(lengths):                                       #
    max_len = torch.max(lengths).item()                                   #
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))    #
    mask = (ids < lengths.unsqueeze(1)).bool()                            #
    return mask                                                           #
###########################################################################

# layers.py
#####################################################################################
class LinearNorm(torch.nn.Module):                                                  # 
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):           #
        super(LinearNorm, self).__init__()                                          #
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)             #
                                                                                    #
        torch.nn.init.xavier_uniform_(                                              #
            self.linear_layer.weight,                                               #
            gain=torch.nn.init.calculate_gain(w_init_gain))                         #
                                                                                    #
    def forward(self, x: Tensor) -> Tensor:                                         #
        return self.linear_layer(x)                                                 #
                                                                                    #
class ConvNorm(torch.nn.Module):                                                    #
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,          #
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):        #
        super(ConvNorm, self).__init__()                                            #
        if padding is None:                                                         #
            assert(kernel_size % 2 == 1)                                            #
            padding = int(dilation * (kernel_size - 1) / 2)                         #
                                                                                    #
        self.conv = torch.nn.Conv1d(in_channels, out_channels,                      #
                                    kernel_size=kernel_size, stride=stride,         #
                                    padding=padding, dilation=dilation,             #
                                    bias=bias)                                      #
                                                                                    #
        torch.nn.init.xavier_uniform_(                                              #
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))       #
                                                                                    #
    def forward(self, signal: Tensor) -> Tensor:                                    #
        conv_signal = self.conv(signal)                                             #
        return conv_signal                                                          #
#####################################################################################

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters: int, attention_kernel_size: int, attention_dim: int):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat: Tensor) -> Tensor:
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention

class Attention(nn.Module):
    def __init__(self, attention_rnn_dim: int, embedding_dim: int, encoder_linear: int, attention_dim: int,
                 attention_location_n_filters=32, attention_location_kernel_size=31):
        super(Attention, self).__init__()
        self.memory_layer   = LinearNorm(embedding_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.query_layer    = LinearNorm(attention_rnn_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.v              = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters, attention_location_kernel_size, attention_dim)
        self.ta             = LinearNorm(embedding_dim + attention_rnn_dim, 1, bias=True)
        
        self.self_memory_layer = LinearNorm(encoder_linear, attention_dim, bias=False, w_init_gain='tanh')
        self.self_query_layer  = LinearNorm(attention_rnn_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.self_v            = LinearNorm(attention_dim, 1, bias=False)
        
        self.score_mask_value = -float(1e20)
        
    def initialize_attn_states(self, memory: Tensor, self_memory: Tensor, mask: Optional[Tensor]=None):
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        
        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.alpha = torch.cat([torch.ones([B, 1]), torch.zeros([B, MAX_TIME])[:, :-1] + 1e-7], dim=1).to(memory.device)
        self.u = (0.5 * torch.ones([B, 1])).to(memory.device)
        
        self.processed_memory = self.memory_layer(memory)
        self.self_processed_memory = self.self_memory_layer(self_memory)
        
        self.mask        = mask
        self.memory      = memory
        self.self_memory = self_memory
        
    def get_alignment_energies(
                               self, 
                               query:                 Tensor, 
                               processed_memory:      Tensor, 
                               attention_weights_cat: Tensor
                               ) -> Tensor:
                               
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))
        energies = energies.squeeze(-1)
        return energies
    
    def forward_attn(
                self, 
                attention_hidden_state: Tensor, 
                bias:                   Optional[float]=None
                ) -> List[Tensor]:
              
        attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1)

        # Do Location Sensitive Attention
        log_energy = self.get_alignment_energies(attention_hidden_state, self.processed_memory, attention_weights_cat)
        
        if self.mask is not None:
            log_energy.data.masked_fill_(self.mask, self.score_mask_value)
        
        # Here using sigmoid not softmax.
        alignment = torch.sigmoid(log_energy) / torch.sigmoid(log_energy).sum(dim=1, keepdim=True)
        self.attention_weights_cum += alignment
        
        # Do Forward Attention
        shift_log_energy = F.pad(self.alpha[:, :-1].clone().to(alignment.device), (1, 0, 0, 0))
        alpha = ((1 - self.u) * self.alpha + self.u * shift_log_energy + 1e-8) * alignment
        alignments = alpha / alpha.sum(dim=1, keepdim=True)
        self.alpha = alignments
        self.attention_weights = alignments
        
        # (Q+K) * V -> output
        attention_context = torch.bmm(alignments.unsqueeze(1), self.memory)
        attention_context = attention_context.squeeze(1)
        
        # Do Transition Agent make alignment appear faster.
        ta_input = torch.cat([attention_context, attention_hidden_state.squeeze(1)], dim=-1)
        self.u   = torch.sigmoid(self.ta(ta_input))
        if bias is not None:
            self.u = self.u * bias

        return [attention_context, alignments]
    
    def additve_attn(self, attention_hidden_state: Tensor) -> List[Tensor]:
        # Do Bahdanau Attention
        processed_query = self.self_query_layer(attention_hidden_state.unsqueeze(1))
        log_energy = self.self_v(torch.tanh(processed_query + self.self_processed_memory))
        log_energy = log_energy.squeeze(-1)
        
        if self.mask is not None:
            log_energy.data.masked_fill_(self.mask, self.score_mask_value)

        # Here using sigmoid not softmax.
        attention_weights = torch.sigmoid(log_energy) / torch.sigmoid(log_energy).sum(dim=1, keepdim=True)
        
        # (Q+K) * V -> output
        attention_context = torch.bmm(attention_weights.unsqueeze(1), self.self_memory)
        attention_context = attention_context.squeeze(1)
        return [attention_context, attention_weights]
    
class Prenet(nn.Module):
    def __init__(self, in_dim: int, sizes: int, spk: int, use_spk_emb: bool):
        super(Prenet, self).__init__()
        if use_spk_emb:
            self.spk_fc1 = LinearNorm(spk, sizes)
            
        self.fc1     = LinearNorm(in_dim, sizes)
        self.fc2     = LinearNorm(sizes, sizes)
        self.fc3     = LinearNorm(sizes, sizes)

    def forward(self, x: Tensor, speaker: Optional[Tensor]=None) -> Tensor:
        # Linear Bottleneck Layer
        x = F.mish(self.fc1(x))
        
        if speaker is not None and Model.add_spk_to_prenet:
            speaker = F.softsign(self.spk_fc1(speaker.squeeze(1))) # Let Speaker Embedding be a bias between [-1, 1].
            x = x + speaker
            
        x = F.dropout(F.mish(self.fc2(x)), drop_rate, training=self.training)
        x = F.dropout(F.mish(self.fc3(x)), drop_rate, training=self.training)
        return x
        
class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, config: Dict = Encoder):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(config.n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(config.embedding,
                         config.embedding,
                         kernel_size=config.kernel_size, stride=1,
                         padding=int((config.kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(config.embedding))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(config.embedding,
                            int(config.embedding / 2), 1,
                            batch_first=True, bidirectional=True)


    def forward(self, x: Tensor, input_lengths: Tensor) -> Tensor:

        if x.size()[0] > 1:
            x_embedded = []
            for b_ind in range(x.size()[0]):  # Speed up
                curr_x = x[b_ind:b_ind+1, :, :input_lengths[b_ind]].clone()
                for conv in self.convolutions:
                    curr_x = F.dropout(F.relu(conv(curr_x)), drop_rate, self.training)
                x_embedded.append(curr_x[0].transpose(0, 1))
            x = torch.nn.utils.rnn.pad_sequence(x_embedded, batch_first=True)
        else:
            for conv in self.convolutions:
                x = F.dropout(F.relu(conv(x)), drop_rate, self.training)
            x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x: Tensor) -> Tensor:
                  
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), drop_rate, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, config: Dict = Decoder):
        super(Decoder, self).__init__()
        self.n_mel_channels    = config.mel_channels
        self.embedding         = config.symbols_embedding
        self.encoder_linear    = config.encoder_linear
        self.attention_rnn_dim = config.attention_rnn_dim
        self.decoder_rnn_dim   = config.decoder_rnn_dim
        self.max_decoder_steps = config.max_decoder_steps
        self.gate_threshold    = config.gate_threshold
        
        if Use_Speaker.use_spk_emb:
            self.embedding      += config.spk_emb_out
            self.encoder_linear += config.spk_emb_out
        
        self.dim = config.decoder_rnn_dim + self.embedding + self.encoder_linear

        self.prenet = Prenet(config.mel_channels, config.prenet_dim, config.spk_emb_out, Use_Speaker.use_spk_emb)

        self.attention_rnn = nn.LSTMCell(
            self.embedding + self.encoder_linear + config.prenet_dim,
            config.attention_rnn_dim
            )
        
        self.attention_layer = Attention(
            config.attention_rnn_dim, self.embedding,
            self.encoder_linear, config.attention_dim
            )
        
        self.decoder_rnn          = nn.LSTMCell(self.dim, config.decoder_rnn_dim, 1)
        self.linear_projection    = LinearNorm(self.dim, config.mel_channels)
        self.gate_layer           = LinearNorm(config.decoder_rnn_dim + config.mel_channels, 1, bias=True, w_init_gain='sigmoid')
        self.multihead_attn       = nn.MultiheadAttention(self.dim, config.num_heads)
        self.gate_bias = 0.0

    def initialize_decoder_states(self, memory: Tensor, self_memory: Tensor):

        # memory        -> Encoder outputs.
        # self_memory -> Another Encoder outputs with using Self-Attention.
        B = memory.size(0)
        MAX_TIME = memory.size(1)
                                             ################
        # Mel -> Prenet -> Concat context -> # Decoder_LSTM # -> Mel_out ----------------
                                             ################                          #|
        concat = torch.cat((memory, self_memory), 2)                                   #|
        self.attention_hidden = Variable(concat.data.new(                              #|
            B, self.attention_rnn_dim).zero_())                                        #| ---- Do          
        self.attention_cell = Variable(concat.data.new(                                #| ----    Attention -> LSTM Decoder -> Linear Output
            B, self.attention_rnn_dim).zero_())                                        #|
                                                                                       #| 
                                                                                       #|
        # Text Concat Mel_out -----------------------------------------------------------
        self.attention_context = Variable(memory.data.new(
            B, self.embedding).zero_())
        self.self_attention_context = Variable(self_memory.data.new(
            B, self.encoder_linear).zero_())
            
                       ##############
        # Attention -> #LSTM Decoder# -> Linear Output
                       ##############
        self.decoder_hidden = Variable(concat.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(concat.data.new(
            B, self.decoder_rnn_dim).zero_())
        
    def decode(
               self, 
               decoder_input: Tensor, 
               bias:          Optional[float]=None
               ) -> List[Tensor]:
    
        # decoder_input -> Mel hidden values after doing Pre-Net.
        # bias          -> Add it to the Forward Attention can let generate mel be faster or slower in the inference.
        # Mel_out concat context to do LSTM
        
        cell_input = torch.cat((decoder_input, self.attention_context, self.self_attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(self.attention_hidden, 0.1, self.training)
        self.attention_cell = F.dropout(self.attention_cell, 0.1, self.training)

        self.attention_context, alignments           = self.attention_layer.forward_attn(self.attention_hidden, bias)
        self.self_attention_context, self_alignments = self.attention_layer.additve_attn(self.attention_hidden)

        # Attention_out concat context to do LSTM       
        decoder_input = torch.cat(
           (self.attention_hidden, self.attention_context, self.self_attention_context), -1)
       
        # Do LSTMCell and Dropout
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, 0.1, self.training)
        
        # LSTM_out_hidden concat context
        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context, self.self_attention_context), dim=1)
        
        # Do Self-Attention and residual connect
        self_attention_in = decoder_hidden_attention_context.unsqueeze(0)
        self_attention_out, _ = self.multihead_attn(self_attention_in, self_attention_in, self_attention_in)
        self_attention_out = self_attention_out.squeeze(0)
        decoder_hidden_attention_context = decoder_hidden_attention_context + self_attention_out

        # Do Linear to generate one frame-level mel, there Linear Layer like LSTM output
        decoder_output = self.linear_projection(decoder_hidden_attention_context)
        
        gate_input = torch.cat((self.decoder_hidden, decoder_output), dim=1)
        gate_prediction = self.gate_layer(gate_input)
        gate_prediction += self.gate_bias
            
        if not self.training and alignments.argmax() == alignments.size(-1) - 1:
            self.gate_bias += 0.1
            
        return [decoder_output, gate_prediction, alignments, self_alignments]

    def forward(
                self, 
                memory:         Tensor, 
                self_memory:    Tensor, 
                decoder_inputs: Tensor, 
                memory_lengths: Tensor,
                speaker:        Optional[Tensor]=None, 
                ) -> List[Tensor]:
                
        # memory         -> Encoder output.
        # self_memory  -> Another Encoder output with using Self-Attention.
        # deocder_inputs -> Target Mels for Training.
        # memory_lengths -> Encoder output lengths.
        # speaker        -> Speaker Embedding.
        
        # Create an zeros matrix [1, batch, 80] for first training frame.
        decoder_input = Variable(memory.data.new(
            memory.size(0), self.n_mel_channels).zero_()).unsqueeze(0)
        
        # Transpose a matrix to [T, batch, 80]. 
        # Notice transpose is not contiguous.
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.transpose(0, 1) # Permute size to [T, B, C].
        
        # Concat the matrix [1+T, batch, 80].
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)

        # (Mel + Speaker Embedding) do prenet.
        decoder_inputs = self.prenet(decoder_inputs, speaker)
        
        # Initialize decoder states, e.g. rnn_hidden, rnn_cell.
        self.initialize_decoder_states(
            memory, self_memory)
        self.attention_layer.initialize_attn_states(memory, self_memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments, self_alignments = [], [], [], []

        while len(mel_outputs) < decoder_inputs.size(0) - 1:

            decoder_input = decoder_inputs[len(mel_outputs)]
            
            mel_output, gate_output, attention_weights, self_attention_weights = self.decode(
                decoder_input)
                
            mel_outputs     += [mel_output.squeeze(1)]
            gate_outputs    += [gate_output.squeeze(1)]
            alignments      += [attention_weights]
            self_alignments += [self_attention_weights]
                
        # Let [List] to be a Multi-Dim Tensor.
        mel_outputs     = torch.stack(mel_outputs).transpose(0, 1).transpose(1, 2)
        gate_outputs    = torch.stack(gate_outputs).transpose(0, 1).contiguous()
        alignments      = torch.stack(alignments).transpose(0, 1)
        self_alignments = torch.stack(self_alignments).transpose(0, 1)

        return mel_outputs, gate_outputs, alignments, self_alignments

    def inference(
                  self, 
                  memory:        Tensor, 
                  self_memory: Tensor, 
                  speaker:       Optional[Tensor]=None, 
                  bias:          Optional[float]=None
                  ) -> List[Tensor]:
    
        # memory         -> Encoder output.
        # self_memory  -> Another Encoder output with using Self-Attention.
        # speaker        -> Speaker Embedding.
        # bias           -> Add it to the Forward Attention can let generate mel be faster or slower in the inference.

        # Create an zeros matrix [batch, 80] for first inference frame.
        decoder_input = Variable(memory.data.new(
            memory.size(0), self.n_mel_channels).zero_())
           
        # Initialize decoder states, e.g. rnn_hidden, rnn_cell. 
        self.initialize_decoder_states(memory, self_memory)
        self.attention_layer.initialize_attn_states(memory, self_memory)

        mel_outputs, gate_outputs, alignments, self_alignments = [], [], [], []
        while True:
            # Do prenet.
            decoder_input = self.prenet(decoder_input, speaker)
            
            mel_output, gate_output, alignment, self_alignment = self.decode(decoder_input)

            mel_outputs     += [mel_output.squeeze(1)]
            gate_outputs    += [gate_output]
            alignments      += [alignment]
            self_alignments += [self_alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                break

            decoder_input = mel_output

        # Let [List] to be a Multi-Dim Tensor.
        mel_outputs     = torch.stack(mel_outputs).transpose(0, 1).transpose(1, 2)
        gate_outputs    = torch.stack(gate_outputs).transpose(0, 1).contiguous()
        alignments      = torch.stack(alignments).transpose(0, 1)
        self_alignments = torch.stack(self_alignments).transpose(0, 1)

        return mel_outputs, gate_outputs, alignments, self_alignments

class Tacotron2(nn.Module):
    def __init__(self, symbols: int, config: Dict = Model):
        super(Tacotron2, self).__init__()
        self.mask_padding   = config.mask_padding
        self.n_mel_channels = config.mel_channels
        self.use_spk_emb    = Use_Speaker.use_spk_emb
        self.use_spk_table  = Use_Speaker.use_spk_table
        
        self.embedding = nn.Embedding(symbols + 1, config.symbols_embedding)
        std = sqrt(2.0 / (symbols + 1 + config.symbols_embedding))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        #mine
        self.encoder_proj   = LinearNorm(config.symbols_embedding, config.encoder_linear)
        self.multihead_attn = nn.MultiheadAttention(config.encoder_linear, config.num_heads)
        
        if self.use_spk_emb:
            if self.use_spk_table: # If True, use Speaker ID Table.
                self.spk_proj = nn.Embedding(config.n_spk + 1, config.spk_emb_out)
            else:                    # If False, use external Speaker Embedding.
                self.spk_proj = LinearNorm(config.spk_emb_in, config.spk_emb_out)
                
    def forward(self, inputs: Tuple) -> List[Tensor]:
        if self.use_spk_emb:
            text_inputs, text_lengths, mels, max_len, output_lengths, speaker = inputs
        else:
            text_inputs, text_lengths, mels, max_len, output_lengths, speaker = inputs
            speaker = None

        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        
        # Fit the Speaker Embedding with Model.
        if self.use_spk_emb:
            if not self.use_spk_table:
                speaker = speaker.float()
                
            speaker = speaker.unsqueeze(1)
            speaker = self.spk_proj(speaker).float()
            
            if len(speaker.shape) > 3:
                speaker = speaker.view(speaker.size(0), 1, -1)

        # Text -> Embedding -> Encoder
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        
        # Another output with using Self-Attention
        self_attn_in     = self.encoder_proj(encoder_outputs)
        self_attn_out    = self_attn_in.permute(1, 0, 2)
        self_attn_out, _ = self.multihead_attn(self_attn_out, self_attn_out, self_attn_out)
        self_attn_out    = self_attn_out.permute(1, 0, 2)
        self_attn_out    = self_attn_out + self_attn_in
        
        # Concat Encoder_ouputs and Encoder_outputs_proj with Speaker embedding
        if self.use_spk_emb:
            spk_emb = speaker.repeat(1, encoder_outputs.size(1), 1)
            encoder_outputs = torch.cat((encoder_outputs, spk_emb), 2)
            self_attn_out = torch.cat((self_attn_out, spk_emb), 2)
	# Decoder
        outputs = self.decoder(
            encoder_outputs, self_attn_out, mels, text_lengths, speaker)
        
        # Fill Mask
        mask = ~get_mask_from_lengths(output_lengths)
        mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2)
        outputs[0].data.masked_fill_(mask, 0.0)
        outputs[1].data.masked_fill_(mask[:, 0, :], 1e3) # gate energy

        return outputs
        
    @torch.no_grad()
    def inference(
                  self, 
                  inputs:  Tensor, 
                  speaker: Optional[Tensor]=None, 
                  bias:    Optional[float]=None
                  ) -> List[Tensor]:
    
        # Fit the Speaker Embedding with Model.
        if self.use_spk_emb:
            if not self.use_spk_table:
                speaker = speaker.float()
                
            speaker = speaker.unsqueeze(0)
            speaker = self.spk_proj(speaker).float()
            
            if len(speaker.shape) > 3:
                speaker = speaker.view(speaker.size(0), 1, -1)
        
        # Text -> Embedding -> Encoder
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        
        # Another output with using Self-Attention        
        self_attn_in     = self.encoder_proj(encoder_outputs)
        self_attn_out    = self_attn_in.permute(1, 0, 2)
        self_attn_out, _ = self.multihead_attn(self_attn_out, self_attn_out, self_attn_out)
        self_attn_out    = self_attn_out.permute(1, 0, 2)
        self_attn_out    = self_attn_out + self_attn_in
        
        # Concat Encoder_ouputs and Encoder_outputs_proj with Speaker embedding
        if self.use_spk_emb:
            spk_emb = speaker.repeat(1, encoder_outputs.size(1), 1)
            encoder_outputs = torch.cat((encoder_outputs, spk_emb), 2)
            self_attn_out = torch.cat((self_attn_out, spk_emb), 2)
        
        # Decoder
        outputs = self.decoder.inference(encoder_outputs, self_attn_out, speaker, bias)
        
        return outputs
