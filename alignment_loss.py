import torch

class ForwardSumLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(ForwardSumLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        self.CTCLoss = torch.nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_logprob, text_lens, mel_lens):
        """
        Args:
        attn_logprob -> [batch, 1, max(mel_lens), max(text_lens)]
        text_lens    -> [lens]
        mel_lens     -> [lens]
        """
        # The CTC loss module assumes the existence of a blank token
        # that can be optionally inserted anywhere in the sequence for
        # a fixed probability.
        # A row must be added to the attention matrix to account for this
        attn_logprob_pd = torch.nn.functional.pad(input=attn_logprob,
                                                  pad=(1, 0, 0, 0, 0, 0, 0, 0),
                                                  value=self.blank_logprob)
        cost_total = 0.0
        
        # for-loop over batch because of variable-length
        # sequences
        
        for bid in range(attn_logprob.shape[0]):
        # construct the target sequence. Every
        # text token is mapped to a unique sequence number,
        # thereby ensuring the monotonicity constraint
        
            target_seq = torch.arange(1, text_lens[bid]+1)
            target_seq = target_seq.unsqueeze(0)
            curr_logprob  = attn_logprob_pd[bid].permute(1, 0, 2)
            curr_log_prob = curr_logprob[:mel_lens[bid],:,:text_lens[bid]+1]
            curr_logprob  = self.log_softmax(curr_logprob[None])[0]
        
            cost = self.CTCLoss(curr_logprob,
                                target_seq,
                                input_lengths=mel_lens[bid:bid+1],
                                target_lengths=text_lens[bid:bid+1])
            cost_total += cost
            
        # average cost over batch
        cost_total = cost_total/attn_logprob.shape[0]
        return cost_total
