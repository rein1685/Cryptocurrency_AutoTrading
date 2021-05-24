# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/107b_models.RNN_FCNPlus.ipynb (unless otherwise specified).

__all__ = ['RNN_FCNPlus', 'LSTM_FCNPlus', 'GRU_FCNPlus', 'MRNN_FCNPlus', 'MLSTM_FCNPlus', 'MGRU_FCNPlus']

# Cell
from ..imports import *
from .layers import *

# Cell
class _RNN_FCN_BasePlus(nn.Sequential):
    def __init__(self, c_in, c_out, seq_len=None, hidden_size=100, rnn_layers=1, bias=True, cell_dropout=0, rnn_dropout=0.8, bidirectional=False, shuffle=True,
                 fc_dropout=0., conv_layers=[128, 256, 128], kss=[7, 5, 3], se=0):

        if shuffle: assert seq_len is not None, 'need seq_len if shuffle=True'

        backbone = _RNN_FCN_Base_Backbone(self._cell, c_in, c_out, seq_len=seq_len, hidden_size=hidden_size, rnn_layers=rnn_layers, bias=bias,
                                          cell_dropout=cell_dropout, rnn_dropout=rnn_dropout, bidirectional=bidirectional, shuffle=shuffle,
                                          conv_layers=conv_layers, kss=kss, se=se)

        head_layers = [nn.Dropout(fc_dropout)] if fc_dropout else []
        head_layers += [nn.Linear(hidden_size * (1 + bidirectional) + conv_layers[-1], c_out)]
        head = nn.Sequential(*head_layers)

        layers = OrderedDict([('backbone', backbone), ('head', head)])
        super().__init__(layers)


class _RNN_FCN_Base_Backbone(Module):
    def __init__(self, _cell, c_in, c_out, seq_len=None, hidden_size=100, rnn_layers=1, bias=True, cell_dropout=0, rnn_dropout=0.8, bidirectional=False,
                 shuffle=True, conv_layers=[128, 256, 128], kss=[7, 5, 3], se=0):

        # RNN - first arg is usually c_in. Authors modified this to seq_len by not permuting x. This is what they call shuffled data.
        self.rnn = _cell(seq_len if shuffle else c_in, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True,
                              dropout=cell_dropout, bidirectional=bidirectional)
        self.rnn_dropout = nn.Dropout(rnn_dropout) if rnn_dropout else noop
        self.shuffle = Permute(0,2,1) if not shuffle else noop # You would normally permute x. Authors did the opposite.

        # FCN
        assert len(conv_layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, conv_layers[0], kss[0])
        self.se1 = SqueezeExciteBlock(conv_layers[0], se) if se != 0 else noop
        self.convblock2 = ConvBlock(conv_layers[0], conv_layers[1], kss[1])
        self.se2 = SqueezeExciteBlock(conv_layers[1], se) if se != 0 else noop
        self.convblock3 = ConvBlock(conv_layers[1], conv_layers[2], kss[2])
        self.gap = GAP1d(1)

        # Common
        self.concat = Concat()

    def forward(self, x):
        # RNN
        rnn_input = self.shuffle(x) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        output, _ = self.rnn(rnn_input)
        last_out = output[:, -1] # output of last sequence step (many-to-one)
        last_out = self.rnn_dropout(last_out)

        # FCN
        x = self.convblock1(x)
        x = self.se1(x)
        x = self.convblock2(x)
        x = self.se2(x)
        x = self.convblock3(x)
        x = self.gap(x)

        # Concat
        x = self.concat([last_out, x])
        return x


class RNN_FCNPlus(_RNN_FCN_BasePlus):
    _cell = nn.RNN

class LSTM_FCNPlus(_RNN_FCN_BasePlus):
    _cell = nn.LSTM

class GRU_FCNPlus(_RNN_FCN_BasePlus):
    _cell = nn.GRU

class MRNN_FCNPlus(_RNN_FCN_BasePlus):
    _cell = nn.RNN
    def __init__(self, *args, se=16, **kwargs):
        super().__init__(*args, se=16, **kwargs)

class MLSTM_FCNPlus(_RNN_FCN_BasePlus):
    _cell = nn.LSTM
    def __init__(self, *args, se=16, **kwargs):
        super().__init__(*args, se=16, **kwargs)

class MGRU_FCNPlus(_RNN_FCN_BasePlus):
    _cell = nn.GRU
    def __init__(self, *args, se=16, **kwargs):
        super().__init__(*args, se=16, **kwargs)