"""
@author Isaac Zheng

This is an implementation of the model described in the paper LONG DOCUMENT SUMMARIZATION WITH TOP-DOWN AND BOTTOM-UP INFERENCE
This model currently (Apr 28) has the highest score at the arXiv and Pubmed dataset and performs well at long document summarization.
You can find this paper at https://doi.org/10.48550/arXiv.2203.07586.
Their model is not open source so this is just my implementation.

My work is based on the Youtube video "Transformer Implementation from Scratch with PyTorch (Attention Is All You Need)!", thanks
Ahmad Chalhoub for his work.
Link to video: https://www.youtube.com/watch?v=f7TnuO02DjM&t=8s

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributed.tensor.parallel import parallelize_module, PairwiseParallel

from utils import utils

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)
    

"""
This is a simple version of multi-head attention used in transformer
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=8):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, mask, cache=None):
        original_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)                  # [batch_size, head_size, q_len, d_k]
        v = v.transpose(1, 2)                  # [batch_size, head_size, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [batch_size, head_size, d_k, k_len]

        # Scaled Dot-Product Attention as in the paper
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))*V
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        x.masked_fill_(mask.unsqueeze(1), -1e9)
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == original_q_size
        return x
    

"""
This feed forward network implementation is the same as that in the transformer
"""
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
    

"""
A simple implementation of local self attention, where each token only attends tokens within a local 
fixed-length window, and thus the complexity does not grow as a function of the input sequence length.
"""
class LocalSelfAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=8):
        super(LocalSelfAttention, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
    
    def self_attend(self, x, mask):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, mask)
        y = self.self_attention_dropout(y)
        x = x + y
        return x

    def forward(self, inputs, mask):
        segments = torch.split(inputs, 512)
        # I hope to compute matrix multiplication in parallel here, but I don't know if it will work.
        # So I commented it out for now.
        # pool = parallelize_module(self.self_attend, PairwiseParallel())
        # inferred_segments = pool.map(self.self_attend, segments)
        inferred_segments = map(lambda x: self.self_attend(x, mask), segments)
        inferred_token_representation = torch.cat(inferred_segments, dim=1)
        return inferred_token_representation
    

"""
implementation of full self attention in the paper
Just a wrap up of multi-head attention
"""
class FullSelfAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=8):
        super(FullSelfAttention, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, mask)
        y = self.self_attention_dropout(y)
        x = x + y
        return x
    
    
"""
This is the polling layer used in bottom-up-inference
The paper mentioned 2 methods, average pooling and adaptive pooling.
For ease of implementation, I am using average pooling
"""
class PoolingLayer(nn.Module):
    def __init__(self, kernel_size, stride):
        super(FeedForwardNetwork, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # infer the output size based on kernel_size and stride
        output_size = (x.size(1) - self.kernel_size) // self.stride
        result = torch.zeros(x.size(0), output_size)
        # I am using average pooling (AvgPool) and hence p = 1 / k, where k is the output size
        for j in range(output_size):
            for n in range(self.kernel_size):
                result[:, j] = (1 / output_size) * x[:, j * self.stride + n]
        return result
    


class BottomUpInference(nn.Module):
    def __init__(self, hidden_size, dropout_rate, N1, N2):
        super(BottomUpInference, self).__init__()

        local_attentions = [LocalSelfAttention(hidden_size, dropout_rate)
                            for _ in range(N1)]

        self.local_attention_layers = nn.ModuleList(local_attentions)

        self.pooling_layer = PoolingLayer(kernel_size=256, stride=192)

        full_attentions = [FullSelfAttention(hidden_size, dropout_rate)
                            for _ in range(N2)]

        self.full_attention_layers = nn.ModuleList(full_attentions)

    def forward(self, inputs, mask):
        inferred_token_representation = inputs
        for local_attention_layer in self.local_attention_layers:
            inferred_token_representation = local_attention_layer(inferred_token_representation, mask)
        
        top_level_representation = self.pooling_layer(inferred_token_representation)

        for full_attention_layer in self.full_attention_layers:
            top_level_representation = full_attention_layer(top_level_representation, mask)

        return inferred_token_representation, top_level_representation
    

class TopDownInferenceLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(TopDownInferenceLayer, self).__init__()

        self.local_attention = LocalSelfAttention(hidden_size, dropout_rate)

        self.tok_seg_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.tok_seg_attention_dropout = nn.Dropout(dropout_rate)
        self.tok_seg_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x, top_level_rep, i_mask):
        y = self.self_attention(x, x, x, i_mask)

        if top_level_rep is not None:
            y = self.tok_seg_attention_norm(x)
            y = self.tok_seg_attention(y, top_level_rep, top_level_rep, i_mask)
            y = self.tok_seg_attention_dropout(y)
            x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
    

"""
implementation of top-down inference layer, putting everything above together
"""
class TopDownInference(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, N3):
        super(TopDownInference, self).__init__()

        top_down_inf_layers = [TopDownInferenceLayer(hidden_size, filter_size, dropout_rate)
                    for _ in range(N3)]
        self.layers = nn.ModuleList(top_down_inf_layers)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, bot_up_inf_tok_rep, top_level_rep, i_mask):
        output = bot_up_inf_tok_rep
        for i, layer in enumerate(self.layers):
            output = layer(output, top_level_rep, i_mask)
        return self.last_norm(output)
    

"""
I used the encoder and decoder of transformer to perform the role of "decoder" in the paper
"""
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):  # pylint: disable=arguments-differ
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.enc_dec_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.enc_dec_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, self_mask, i_mask, cache):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, self_mask)
        y = self.self_attention_dropout(y)
        x = x + y

        if enc_output is not None:
            y = self.enc_dec_attention_norm(x)
            y = self.enc_dec_attention(y, enc_output, enc_output, i_mask,
                                       cache)
            y = self.enc_dec_attention_dropout(y)
            x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, N4):
        super(Encoder, self).__init__()

        encoders = [EncoderLayer(hidden_size, filter_size, dropout_rate)
                    for _ in range(N4)]
        self.layers = nn.ModuleList(encoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, inputs, mask):
        encoder_output = inputs
        for enc_layer in self.layers:
            encoder_output = enc_layer(encoder_output, mask)
        return self.last_norm(encoder_output)


class Decoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, N4):
        super(Decoder, self).__init__()

        decoders = [DecoderLayer(hidden_size, filter_size, dropout_rate)
                    for _ in range(N4)]
        self.layers = nn.ModuleList(decoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, targets, enc_output, i_mask, t_self_mask, cache):
        decoder_output = targets
        for i, dec_layer in enumerate(self.layers):
            layer_cache = None
            if cache is not None:
                if i not in cache:
                    cache[i] = {}
                layer_cache = cache[i]
            decoder_output = dec_layer(decoder_output, enc_output,
                                       t_self_mask, i_mask, layer_cache)
        return self.last_norm(decoder_output)
    

"""
Puts everything together, wraps up everything
"""
class TDBU(nn.Module):
    def __init__(self, i_vocab_size, t_vocab_size,
                 N1=6,
                 N2=6,
                 N3=6,
                 N4=6,
                 hidden_size=512,
                 filter_size=2048,
                 dropout_rate=0.1,
                 share_target_embedding=True,
                 has_inputs=True,
                 src_pad_idx=None,
                 trg_pad_idx=None):
        super(TDBU, self).__init__()

        self.hidden_size = hidden_size
        self.emb_scale = hidden_size ** 0.5
        self.has_inputs = has_inputs
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.t_vocab_embedding = nn.Embedding(t_vocab_size, hidden_size)
        nn.init.normal_(self.t_vocab_embedding.weight, mean=0,
                        std=hidden_size**-0.5)
        self.t_emb_dropout = nn.Dropout(dropout_rate)
        self.decoder = Decoder(hidden_size, filter_size,
                               dropout_rate, N4)

        if has_inputs:
            if not share_target_embedding:
                self.i_vocab_embedding = nn.Embedding(i_vocab_size,
                                                      hidden_size)
                nn.init.normal_(self.i_vocab_embedding.weight, mean=0,
                                std=hidden_size**-0.5)
            else:
                self.i_vocab_embedding = self.t_vocab_embedding

            self.i_emb_dropout = nn.Dropout(dropout_rate)

            self.bottom_up_inf = BottomUpInference(hidden_size, dropout_rate, N1, N2)
            
            self.top_down_inf = TopDownInference(hidden_size, filter_size, dropout_rate, N3)
            
            self.encoder = Encoder(hidden_size, filter_size,
                                   dropout_rate, N4)

        # For positional encoding
        num_timescales = self.hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

    def forward(self, inputs, targets):
        enc_output, i_mask = None, None
        if self.has_inputs:
            i_mask = utils.create_pad_mask(inputs, self.src_pad_idx)
            inputs_inf = self.topDownBottomUpInference(inputs, i_mask)
            enc_output = self.encode(inputs_inf, i_mask)

        t_mask = utils.create_pad_mask(targets, self.trg_pad_idx)
        target_size = targets.size()[1]
        t_self_mask = utils.create_trg_self_mask(target_size,
                                                 device=targets.device)
        return self.decode(targets, enc_output, i_mask, t_self_mask, t_mask)
    
    def topDownBottomUpInference(self, inputs, i_mask):
        # Input embedding
        input_embedded = self.i_vocab_embedding(inputs)
        input_embedded.masked_fill_(i_mask.squeeze(1).unsqueeze(-1), 0)
        input_embedded *= self.emb_scale
        input_embedded += self.get_position_encoding(inputs)
        input_embedded = self.i_emb_dropout(input_embedded)

        inferred_token_representation, top_level_representation = self.bottom_up_inf(input_embedded, i_mask)

        return self.top_down_inf(inferred_token_representation, top_level_representation, i_mask)

    def encode(self, inputs, i_mask):
        return self.encoder(inputs, i_mask)

    def decode(self, targets, enc_output, i_mask, t_self_mask, t_mask,
               cache=None):
        # target embedding
        target_embedded = self.t_vocab_embedding(targets)
        target_embedded.masked_fill_(t_mask.squeeze(1).unsqueeze(-1), 0)

        # Shifting
        target_embedded = target_embedded[:, :-1]
        target_embedded = F.pad(target_embedded, (0, 0, 1, 0))

        target_embedded *= self.emb_scale
        target_embedded += self.get_position_encoding(targets)
        target_embedded = self.t_emb_dropout(target_embedded)

        # decoder
        decoder_output = self.decoder(target_embedded, enc_output, i_mask,
                                      t_self_mask, cache)
        # linear
        output = torch.matmul(decoder_output,
                              self.t_vocab_embedding.weight.transpose(0, 1))

        return output

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)
        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        signal = signal.view(1, max_length, self.hidden_size)
        return signal