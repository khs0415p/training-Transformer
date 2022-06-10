import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

class Embeddings(nn.Module):
    def __init__(self, vocab_size, emb_dim, PAD_IDX):
        super(Embeddings, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.PAD_IDX = PAD_IDX
        self.emb_layer = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=self.PAD_IDX)

    def forward(self, x):
        out = self.emb_layer(x)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, emb_dim):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim

        self.pe = torch.zeros(self.max_len, self.emb_dim)
        self.pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.emb_dim, 2) * -(math.log(10000.0)/self.emb_dim))

        self.pe[:, 0::2] = torch.sin(self.pos * div_term)
        self.pe[:, 1::2] = torch.cos(self.pos * div_term)

        self.pe = nn.Parameter(self.pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_head, bias):
        super(MultiHeadAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.d_k = emb_dim // self.num_head
        self.bias = bias

        self.query_layer = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)
        self.key_layer = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)
        self.value_layer = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)
        self.fc_layer = nn.Linear(self.emb_dim, self.emb_dim, bias=self.bias)

    def scaled_dot_product(self, query, key, value, mask):
        attention_score = torch.matmul(query, key.transpose(-1,-2)) / math.sqrt(self.emb_dim)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, float('-inf'))

        attention_prob = F.softmax(attention_score, dim=-1)
        attention_score = torch.matmul(attention_prob, value)

        return attention_prob, attention_score

    def transform(self, x, layer):
        out = layer(x)
        out = out.view(self.batch_size, -1, self.num_head, self.d_k)
        out = out.transpose(1,2)

        return out

    def forward(self, query, key, value, mask):
        self.batch_size = query.size(0)

        query = self.transform(query, self.query_layer)
        key = self.transform(key, self.key_layer)
        value = self.transform(value, self.value_layer)

        if mask is not None:
            mask = mask.unsqueeze(1)

        attn_prob, attn_score = self.scaled_dot_product(query, key, value, mask)
        attn_score = attn_score.transpose(1,2).contiguous().view(self.batch_size, -1, self.emb_dim)
        attn_score = self.fc_layer(attn_score)

        return attn_prob, attn_score

class PositionWiseFeedForward(nn.Module):
    def __init__(self, emb_dim, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.emb_dim = emb_dim
        self.dropout = dropout

        self.FFN_1 = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.FFN_2 = nn.Sequential(
            nn.Linear(self.emb_dim * 4, self.emb_dim)
        )

    def forward(self, x):
        output = self.FFN_1(x)
        output = self.FFN_2(output)

        return output

class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, num_head, bias, dropout):
        super(EncoderLayer, self).__init__()
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.bias = bias
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.norm_layer = nn.LayerNorm(self.emb_dim, eps=1e-6)

        self.multi_attention = MultiHeadAttention(self.emb_dim, self.num_head, self.bias)
        self.position_feed_forward = PositionWiseFeedForward(self.emb_dim, self.dropout)

    def forward(self, x, mask):
        attn_prob, attn_score = self.multi_attention(query=x, key=x, value=x, mask=mask)
        output = self.dropout_layer(attn_score)
        output = self.norm_layer(x + output)

        x = output
        output = self.position_feed_forward(output)
        output = self.dropout_layer(output)
        output = self.norm_layer(x + output)

        return attn_prob, output


class Encoder(nn.Module):
    def __init__(self, vocab_size, enc_N, emb_dim, num_head, max_len, bias, dropout, PAD_IDX):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.enc_N = enc_N
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.max_len = max_len
        self.bias = bias
        self.dropout = dropout
        self.PAD_IDX = PAD_IDX

        self.dropout_layer = nn.Dropout(self.dropout)
        self.emb_layer = Embeddings(self.vocab_size, self.emb_dim, self.PAD_IDX)
        self.pe_layer = PositionalEncoding(self.max_len, self.emb_dim)
        self.encoders = nn.ModuleList([EncoderLayer(self.emb_dim, self.num_head, self.bias, self.dropout) for _ in range(self.enc_N)])

    def forward(self, x, mask=None):
        output = self.emb_layer(x) + self.pe_layer(x)
        output = self.dropout_layer(output)

        all_prob = []
        for encoder in self.encoders:
            attn_prob, output = encoder(output, mask)
            all_prob.append(attn_prob.detach().cpu())

        return all_prob, output



class DecoderLayer(nn.Module):
    def __init__(self, emb_dim, num_head, bias, dropout):
        super(DecoderLayer, self).__init__()
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.bias = bias
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.norm_layer = nn.LayerNorm(self.emb_dim, eps=1e-6)

        self.masked_attention = MultiHeadAttention(self.emb_dim, self.num_head, self.bias)
        self.enc_dec_attention = MultiHeadAttention(self.emb_dim, self.num_head, self.bias)
        self.position_feed_forward = PositionWiseFeedForward(self.emb_dim, self.dropout)

    def forward(self, x, enc_output, dec_mask, enc_mask):
        dec_prob, output = self.masked_attention(query=x, key=x, value=x, mask=dec_mask)
        output = self.dropout_layer(output)
        output = self.norm_layer(x + output)

        x = output
        dec_enc_prob, output = self.enc_dec_attention(query=x, key=enc_output, value=enc_output, mask=enc_mask)
        output = self.dropout_layer(output)
        output = self.norm_layer(x + output)

        x = output
        output = self.position_feed_forward(output)
        output = self.dropout_layer(output)
        output = self.norm_layer(x + output)

        return dec_prob, dec_enc_prob, output

class Decoder(nn.Module):
    def __init__(self, vocab_size, dec_N, emb_dim, num_head, max_len, bias, dropout, PAD_IDX):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.dec_N = dec_N
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.max_len = max_len
        self.bias = bias
        self.dropout = dropout
        self.PAD_IDX = PAD_IDX

        self.dropout_layer = nn.Dropout(self.dropout)
        self.emb_layer = Embeddings(self.vocab_size, self.emb_dim, self.PAD_IDX)
        self.pe_layer = PositionalEncoding(self.max_len, self.emb_dim)
        self.decoders = nn.ModuleList([DecoderLayer(self.emb_dim, self.num_head, self.bias, self.dropout) for _ in range(self.dec_N)])

    def forward(self, x, enc_output, dec_mask, enc_mask):
        output = self.emb_layer(x) + self.pe_layer(x)
        output = self.dropout_layer(output)

        all_prob, all_dec_enc_prob = [], []
        for decoder in self.decoders:
            dec_prob, dec_enc_prob, output = decoder(output, enc_output, dec_mask, enc_mask)
            all_prob.append(dec_prob.detach().cpu())
            all_dec_enc_prob.append(dec_enc_prob.detach().cpu())

        return all_dec_enc_prob, output

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.device = self.config.device
        self.vocab_size = self.config.src_vocab_size
        self.enc_N = self.config.enc_N
        self.dec_N = self.config.dec_N
        self.emb_dim = self.config.d_embed
        self.num_head = self.config.head
        self.max_len = self.config.max_len
        self.bias = bool(self.config.bias)
        self.dropout = self.config.dropout
        self.PAD_IDX = self.config.pad_id

        self.encoder = Encoder(self.vocab_size, self.enc_N, self.emb_dim, self.num_head, self.max_len, self.bias, self.dropout, self.PAD_IDX)
        self.decoder = Decoder(self.vocab_size, self.dec_N, self.emb_dim, self.num_head, self.max_len, self.bias, self.dropout, self.PAD_IDX)
        self.FC_layer = nn.Linear(self.emb_dim, self.vocab_size)

    def make_mask(self, src, trg):
        src_mask = (src != self.PAD_IDX).unsqueeze(-2)

        trg_mask = (trg != self.PAD_IDX).unsqueeze(-2)
        subsequent_mask = np.triu(np.ones((1, trg_mask.size(-1), trg_mask.size(-1))), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask) == 0

        trg_mask = trg_mask & Variable(subsequent_mask.type_as(trg_mask.data))

        return src_mask, trg_mask.to(self.device)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, src, trg):
        enc_mask, dec_mask = self.make_mask(src, trg)

        all_enc_prob, enc_output = self.encoder(src, enc_mask)
        all_enc_dec_prob, dec_output = self.decoder(trg, enc_output, dec_mask, enc_mask)
        output = self.FC_layer(dec_output)

        return all_enc_dec_prob, output
