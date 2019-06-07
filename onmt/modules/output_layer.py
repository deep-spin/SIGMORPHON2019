import torch
import torch.nn as nn


class MultilingualBias(nn.Embedding):

    def reset_parameters(self):
        nn.init.uniform_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)


class OutputLayer(nn.Module):

    def __init__(self, hidden_size, vocab_size, transform, bias_vectors=1):
        super(OutputLayer, self).__init__()
        self.weight_matrix = nn.Linear(
            hidden_size, vocab_size, bias=bias_vectors == 1)
        if bias_vectors > 1:
            self.multi_bias = MultilingualBias(bias_vectors, vocab_size)
        else:
            self.multi_bias = None
        self.transform = transform

    def forward(self, hidden, language=None, transform=False, **kwargs):
        out = self.weight_matrix(hidden)
        if self.multi_bias is not None:
            assert language is not None
            b = self.multi_bias(language)
            out += b
        if transform:
            # used at translation time
            out = self.transform(out)
        return out
