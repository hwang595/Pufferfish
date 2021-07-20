import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


from enum import IntEnum
from typing import Optional, Tuple


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class NaiveLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, layer_index=0):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        # input gate
        self.W_ii = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hi = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = Parameter(torch.Tensor(hidden_sz))
        # forget gate
        self.W_if = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hf = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = Parameter(torch.Tensor(hidden_sz))
        # ???
        self.W_ig = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hg = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_g = Parameter(torch.Tensor(hidden_sz))
        # output gate
        self.W_io = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_ho = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = Parameter(torch.Tensor(hidden_sz))

        print("##### Sizes: W_ii {}, W_hi: {}, W_if: {}, W_hf: {}, W_ig: {}, W_hg: {}, W_io: {}, W_ho: {}".format(
                    self.W_ii.size(), self.W_hi.size(), self.W_if.size(), self.W_hf.size(), 
                    self.W_ig.size(), self.W_hg.size(), self.W_io.size(), self.W_ho.size()))

        self._layer_index = layer_index # for stacked LSTM
        
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        
    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        seq_sz, bs, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_sz): # iterate over the time steps
            x_t = x[t, :, :]

            i_t = torch.nn.functional.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
            f_t = torch.nn.functional.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)
            g_t = torch.nn.functional.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)
            o_t = torch.nn.functional.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.nn.functional.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        #hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()

        return hidden_seq, (h_t, c_t)


class OptimizedLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.weight_ih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        
    def forward(self, x: torch.Tensor, 
                init_states: Optional[Tuple[torch.Tensor]]=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device), 
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
        
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)


class LowRankLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, layer_index=0, rank_ratio=0.25):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        # input gate
        #self.W_ii = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_ii_U = Parameter(torch.Tensor(input_sz, int(input_sz*rank_ratio)))
        self.W_ii_V = Parameter(torch.Tensor(int(input_sz*rank_ratio), hidden_sz))        
        self.W_hi_U = Parameter(torch.Tensor(hidden_sz, int(hidden_sz*rank_ratio)))
        self.W_hi_V = Parameter(torch.Tensor(int(hidden_sz*rank_ratio), hidden_sz))
        self.b_i = Parameter(torch.Tensor(hidden_sz))
        # forget gate
        self.W_if_U = Parameter(torch.Tensor(input_sz, int(input_sz*rank_ratio)))
        self.W_if_V = Parameter(torch.Tensor(int(input_sz*rank_ratio), hidden_sz)) 
        self.W_hf_U = Parameter(torch.Tensor(hidden_sz, int(hidden_sz*rank_ratio)))
        self.W_hf_V = Parameter(torch.Tensor(int(hidden_sz*rank_ratio), hidden_sz))
        self.b_f = Parameter(torch.Tensor(hidden_sz))
        # ???
        self.W_ig_U = Parameter(torch.Tensor(input_sz, int(input_sz*rank_ratio)))
        self.W_ig_V = Parameter(torch.Tensor(int(input_sz*rank_ratio), hidden_sz)) 
        self.W_hg_U = Parameter(torch.Tensor(hidden_sz, int(hidden_sz*rank_ratio)))
        self.W_hg_V = Parameter(torch.Tensor(int(hidden_sz*rank_ratio), hidden_sz))
        self.b_g = Parameter(torch.Tensor(hidden_sz))
        # output gate
        self.W_io_U = Parameter(torch.Tensor(input_sz, int(input_sz*rank_ratio)))
        self.W_io_V = Parameter(torch.Tensor(int(input_sz*rank_ratio), hidden_sz)) 
        self.W_ho_U = Parameter(torch.Tensor(hidden_sz, int(hidden_sz*rank_ratio)))
        self.W_ho_V = Parameter(torch.Tensor(int(hidden_sz*rank_ratio), hidden_sz))
        self.b_o = Parameter(torch.Tensor(hidden_sz))

        self._layer_index = layer_index # for stacked LSTM
        
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        
    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        seq_sz, bs, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_sz): # iterate over the time steps
            x_t = x[t, :, :]

            i_t = torch.nn.functional.sigmoid(x_t @ self.W_ii_U @ self.W_ii_V + h_t @ self.W_hi_U @ self.W_hi_V + self.b_i)
            f_t = torch.nn.functional.sigmoid(x_t @ self.W_if_U @ self.W_if_V  + h_t @ self.W_hf_U @ self.W_hf_V + self.b_f)
            g_t = torch.nn.functional.tanh(x_t @ self.W_ig_U @ self.W_ig_V + h_t @ self.W_hg_U @ self.W_hg_V + self.b_g)
            o_t = torch.nn.functional.sigmoid(x_t @ self.W_io_U @ self.W_io_V + h_t @ self.W_ho_U @ self.W_ho_V + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.nn.functional.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        return hidden_seq, (h_t, c_t)


class StackedLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers

        # multi-layer LSTM
        if num_layers == 2:
            self.lstm1 = NaiveLSTM(input_sz=input_sz, hidden_sz=hidden_sz, layer_index=0)
            self.dropout = nn.Dropout(p=dropout)
            self.lstm2 = NaiveLSTM(input_sz=hidden_sz, hidden_sz=hidden_sz, layer_index=1)
        elif num_layers == 1:
            self.lstm1 = NaiveLSTM(input_sz=input_sz, hidden_sz=hidden_sz, layer_index=0)
        
    def forward(self, x, init_states):
        # """Assumes x is of shape (batch, sequence, feature)"""
        if self.num_layers == 2:
            output, hidden = self.lstm1(x=x, 
                                init_states=init_states)
            output = self.dropout(output)
            output, hidden = self.lstm2(x=output)
        else:
            output, hidden = self.lstm1(x=x, init_states=init_states)
        return output, hidden


class LowRankStackedLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, num_layers, dropout, rank_ratio=0.25):
        super().__init__()
        self.num_layers = num_layers

        # multi-layer LSTM
        if num_layers == 2:
            self.lstm1 = LowRankLSTM(input_sz=input_sz, hidden_sz=hidden_sz, 
                                    layer_index=0, rank_ratio=0.25)
            self.dropout = nn.Dropout(p=dropout)
            self.lstm2 = LowRankLSTM(input_sz=hidden_sz, hidden_sz=hidden_sz, 
                                    layer_index=1, rank_ratio=0.25)
        elif num_layers == 1:
            self.lstm1 = LowRankLSTM(input_sz=input_sz, hidden_sz=hidden_sz, 
                                    layer_index=0, rank_ratio=0.25)
        
    def forward(self, x, init_states):
        # """Assumes x is of shape (batch, sequence, feature)"""
        if self.num_layers == 2:
            output, hidden = self.lstm1(x=x, 
                                init_states=init_states)
            output = self.dropout(output)
            output, hidden = self.lstm2(x=output)
        else:
            output, hidden = self.lstm1(x=x, init_states=init_states)
        return output, hidden


class LowRankRNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, rank_ratio=0.25, tie_weights=False):
        super(LowRankRNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            #self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            #self.rnn = StackedLSTM(ninp, nhid, nlayers, dropout=dropout)
            self.rnn = LowRankStackedLSTM(ninp, nhid, nlayers, dropout=dropout, rank_ratio=rank_ratio)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(bsz, self.nhid),
                    weight.new_zeros(bsz, self.nhid))
        else:
            return weight.new_zeros(bsz, self.nhid)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            #self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            self.rnn = StackedLSTM(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        # weight = next(self.parameters())
        # if self.rnn_type == 'LSTM':
        #     return (weight.new_zeros(self.nlayers, bsz, self.nhid),
        #             weight.new_zeros(self.nlayers, bsz, self.nhid))
        # else:
        #     return weight.new_zeros(self.nlayers, bsz, self.nhid)
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(bsz, self.nhid),
                    weight.new_zeros(bsz, self.nhid))
        else:
            return weight.new_zeros(bsz, self.nhid)



# class RNNModel(nn.Module):
#     """Container module with an encoder, a recurrent module, and a decoder."""

#     def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
#         super(RNNModel, self).__init__()
#         self.ntoken = ntoken
#         self.drop = nn.Dropout(dropout)
#         self.encoder = nn.Embedding(ntoken, ninp)
#         if rnn_type in ['LSTM', 'GRU']:
#             self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
#         else:
#             try:
#                 nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
#             except KeyError:
#                 raise ValueError( """An invalid option for `--model` was supplied,
#                                  options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
#             self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
#         self.decoder = nn.Linear(nhid, ntoken)

#         # Optionally tie weights as in:
#         # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
#         # https://arxiv.org/abs/1608.05859
#         # and
#         # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
#         # https://arxiv.org/abs/1611.01462
#         if tie_weights:
#             if nhid != ninp:
#                 raise ValueError('When using the tied flag, nhid must be equal to emsize')
#             self.decoder.weight = self.encoder.weight

#         self.init_weights()

#         self.rnn_type = rnn_type
#         self.nhid = nhid
#         self.nlayers = nlayers

#     def init_weights(self):
#         initrange = 0.1
#         nn.init.uniform_(self.encoder.weight, -initrange, initrange)
#         nn.init.zeros_(self.decoder.weight)
#         nn.init.uniform_(self.decoder.weight, -initrange, initrange)

#     def forward(self, input, hidden):
#         emb = self.drop(self.encoder(input))
#         output, hidden = self.rnn(emb, hidden)
#         output = self.drop(output)
#         decoded = self.decoder(output)
#         decoded = decoded.view(-1, self.ntoken)
#         return F.log_softmax(decoded, dim=1), hidden

#     def init_hidden(self, bsz):
#         weight = next(self.parameters())
#         if self.rnn_type == 'LSTM':
#             return (weight.new_zeros(self.nlayers, bsz, self.nhid),
#                     weight.new_zeros(self.nlayers, bsz, self.nhid))
#         else:
#             return weight.new_zeros(self.nlayers, bsz, self.nhid)

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
