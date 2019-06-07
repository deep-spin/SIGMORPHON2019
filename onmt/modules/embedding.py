import torch
import torch.nn as nn


# this shouldn't need to exist, but does
class SetEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, mode='sum'):
        super(SetEmbedding, self).__init__()
        self.emb = nn.EmbeddingBag(vocab_size, emb_size, mode=mode)

    def forward(self, input):
        return self.emb(input.transpose(0, 1))

    @property
    def embedding_dim(self):
        return self.emb.embedding_dim


class InflectionLSTMEncoder(nn.Module):
    def __init__(self, embedding, out_size, bidirectional=True, **kwargs):
        super(InflectionLSTMEncoder, self).__init__()
        assert out_size % 2 == 0
        self.embedding = embedding
        self.rnn = nn.LSTM(
            embedding.embedding_dim,
            out_size // 2,
            bidirectional=bidirectional,
            **kwargs)

    def forward(self, input, **kwargs):
        emb = self.embedding(input, **kwargs)
        rnn_out, _ = self.rnn(emb)
        return rnn_out


class MultispaceEmbedding(nn.ModuleDict):
    """
    An embedding module, but with added side information (for example, a
    language embedding).
    """
    def __init__(self, main_emb, mode='token', **kwargs):
        assert mode in ['token', 'feature', None]
        super(MultispaceEmbedding, self).__init__(
            {k: v for k, v in kwargs.items() if v is not None}
        )
        self._order = sorted(k for k in self.keys())
        self['main'] = main_emb

        if mode == 'feature':
            self.embedding_dim = sum(m.embedding_dim for m in self.values())
        else:
            self.embedding_dim = main_emb.embedding_dim
            assert all(m.embedding_dim == self.embedding_dim
                       for m in self.values())

        self.mode = mode

    def forward(self, input, **kwargs):
        """
        input (LongTensor): sequence length x batch size
        additional arguments:
        """
        main_emb = self['main'](input)
        if self.mode is None:
            return main_emb
        seq_len = main_emb.size(0)
        embs = []
        for k in self._order:
            side_info = kwargs[k]
            side_emb = self[k](side_info)
            # but what to do if you have multiple pieces and you're in
            # feature mode?
            if side_emb.dim() == 2:
                side_emb = side_emb.unsqueeze(0)
                if self.mode == 'feature':
                    side_emb = side_emb.expand(seq_len, -1, -1)
            embs.append(side_emb)
        embs.append(main_emb)
        cat_dim = 2 if self.mode == 'feature' else 0
        emb = torch.cat(embs, cat_dim)
        return emb

    @property
    def num_embeddings(self):
        """Vocab size for the main embedding level"""
        return self["main"].num_embeddings
