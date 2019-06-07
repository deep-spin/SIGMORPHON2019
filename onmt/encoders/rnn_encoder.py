import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def fix_brnn_hidden(h):
    # The encoder hidden is  (layers*directions) x batch x dim.
    # We need to convert it to layers x batch x (directions*dim).
    return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)


class RNNEncoder(nn.Module):

    def __init__(self, emb, rnn):
        super(RNNEncoder, self).__init__()

        self.embedding = emb
        self.rnn = rnn

    def forward(self, src, lengths=None, **kwargs):
        emb = self.embedding(src, **kwargs)
        if lengths is not None:
            if emb.size(0) > src.size(0):
                # for lang_rep == token case
                lengths = lengths + 1
            emb = pack(emb, lengths)

        memory_bank, encoder_final = self.rnn(emb)

        if lengths is not None:
            memory_bank = unpack(memory_bank)[0]

        if not isinstance(encoder_final, tuple):
            encoder_final = encoder_final,  # for GRU
        if self.rnn.bidirectional:
            encoder_final = tuple(fix_brnn_hidden(h) for h in encoder_final)

        return encoder_final, memory_bank
