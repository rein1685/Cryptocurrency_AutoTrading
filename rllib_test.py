from model.models.RNN_FCN import MGRU_FCN
import torch

test_input = torch.randn(128, 13, 128)
model = MGRU_FCN(c_in=13, c_out=9, seq_len=128)
pi, v = model(test_input)
print(pi[0])
print(v[0])