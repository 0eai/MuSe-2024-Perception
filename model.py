import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from config import ACTIVATION_FUNCTIONS, device

# device = 'cuda:1'

class RNN(nn.Module):
    def __init__(self, d_in, d_out, n_layers=1, bi=True, dropout=0.2, n_to_1=False, device='cpu'):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(input_size=d_in, hidden_size=d_out, bidirectional=bi, num_layers=n_layers, dropout=dropout)
        self.device = device
        self.n_layers = n_layers
        self.d_out = d_out
        self.n_directions = 2 if bi else 1
        self.n_to_1 = n_to_1

    def forward(self, x, x_len):
        x_packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        rnn_enc = self.rnn(x_packed)

        if self.n_to_1:
            # hiddenstates, h_n, only last layer
            return last_item_from_packed(rnn_enc[0], x_len, self.device)
            # return last_item_from_packed(rnn_enc[0], x_len)

        else:
            x_out = rnn_enc[0]
            x_out = pad_packed_sequence(x_out, total_length=x.size(1), batch_first=True)[0]

        return x_out

#https://discuss.pytorch.org/t/get-each-sequences-last-item-from-packed-sequence/41118/7
def last_item_from_packed(packed, lengths, device):
# def last_item_from_packed(packed, lengths):
    sum_batch_sizes = torch.cat((
        torch.zeros(2, dtype=torch.int64),
        torch.cumsum(packed.batch_sizes, 0)
    )).to(device)
    sorted_lengths = lengths[packed.sorted_indices].to(device)
    last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.size(0)).to(device)
    last_seq_items = packed.data[last_seq_idxs]
    last_seq_items = last_seq_items[packed.unsorted_indices]
    return last_seq_items


class OutLayer(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=.0, bias=.0):
        super(OutLayer, self).__init__()
        self.fc_1 = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(True), nn.Dropout(dropout))
        self.fc_2 = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.fc_2.bias.data, bias)

    def forward(self, x):
        y = self.fc_2(self.fc_1(x))
        return y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.device = device
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)).to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, d_in, d_model, nhead, n_layers=2, dim_feedforward=2048, dropout=0.1, device='cpu'):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.device = device
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, device=device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.d_model = d_model
        self.embedding = nn.Linear(d_in, d_model)

    def forward(self, x, x_len):
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=self.device))
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # Transpose to fit the Transformer input format (S, N, E)
        mask = self.generate_square_subsequent_mask(x.size(0)).to(self.device)
        output = self.transformer_encoder(x, mask)
        output = output.permute(1, 0, 2)  # Transpose back to original format (N, S, E)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x):
        weights = F.softmax(self.attention(x), dim=1)
        output = torch.sum(weights * x, dim=1)
        return output

class ConvPooling(nn.Module):
    def __init__(self, d_model):
        super(ConvPooling, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (N, E, S)
        x = F.relu(self.conv(x))
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
        return x
    
class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        print(params)
        self.inp = nn.Linear(params.d_in, params.model_dim, bias=False)

        d_encoder_out = params.model_dim
        
        if params.encoder == 'RNN':
            self.encoder = RNN(params.model_dim, params.model_dim, n_layers=params.encoder_n_layers, bi=params.rnn_bi,
                           dropout=params.encoder_dropout, n_to_1=params.n_to_1, device=params.device)
            d_encoder_out = params.model_dim * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.model_dim
        elif params.encoder in ['TF', 'TF_AP', 'TF_CP']:
            self.encoder = TransformerModel(params.model_dim, params.model_dim, params.nhead, 
                                        n_layers=params.encoder_n_layers, dim_feedforward=params.dim_feedforward, dropout=params.encoder_dropout, device=params.device)

        self.attention_pooling = AttentionPooling(d_encoder_out)  # Instantiate AttentionPooling
        self.conv_pooling = ConvPooling(d_encoder_out)
        self.out = OutLayer(d_encoder_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def forward(self, x, x_len):
        x = self.inp(x)
        x = self.encoder(x, x_len)
        if ('encoder' in self.params.__dict__.keys() and self.params.encoder == 'TF') or ('model_type' in self.params.__dict__.keys() and self.params.model_type == 'TF'):  # TODO ADDED
            x = torch.mean(x, dim=1) # TODO ADDED
        elif 'encoder' in self.params.__dict__.keys() and self.params.encoder == 'TF_AP':  # TODO ADDED
            x = self.attention_pooling(x)
        elif 'encoder' in self.params.__dict__.keys() and self.params.encoder == 'TF_CP':  # TODO ADDED
            x = self.conv_pooling(x)
        y = self.out(x)
        activation = self.final_activation(y)
        return activation, x

    def set_n_to_1(self, n_to_1):
        self.encoder.n_to_1 = n_to_1