from collections import defaultdict

import torch
import torch.nn as nn

from onmt.utils.misc import aeq


class RNNDecoderBase(nn.Module):

    def __init__(self, emb, rnn, attn, dropout=0.0):
        super(RNNDecoderBase, self).__init__()

        self.embeddings = emb
        self.rnn = rnn
        self.attn = attn
        self.dropout = nn.Dropout(dropout)

        self.state = {}

    def init_state(self, enc_final):
        """ Init decoder state with last state of the encoder """
        self.state["hidden"] = enc_final

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.rnn.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].new_zeros(*h_size).unsqueeze(0)

    def map_state(self, fn):
        self.state["hidden"] = tuple(fn(h, 1) for h in self.state["hidden"])
        self.state["input_feed"] = fn(self.state["input_feed"], 1)

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        self.state["input_feed"] = self.state["input_feed"].detach()


class RNNDecoder(RNNDecoderBase):

    def forward(self, tgt, memory_bank, **kwargs):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[tgt_len x batch]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * dec_outs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # should forward still take memory_lengths as a kwarg in all cases?
        tgt_emb = self.embeddings(tgt, **kwargs)

        attns = {}

        hidden = self.state["hidden"][0] if isinstance(self.rnn, nn.GRU) \
            else self.state["hidden"]
        dec_outs, dec_state = self.rnn(tgt_emb, hidden)

        tgt_len, tgt_batch, _ = tgt_emb.size()
        output_len, output_batch, _ = dec_outs.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)

        dec_outs, attns = self.attn(
            dec_outs.transpose(0, 1).contiguous(), memory_bank, **kwargs
        )

        dec_outs = self.dropout(dec_outs)

        if not isinstance(dec_state, tuple):
            dec_state = dec_state,
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)

        return dec_outs, attns


class InputFeedRNNDecoder(RNNDecoderBase):
    def forward(self, tgt, memory_bank, **kwargs):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[tgt_len x batch]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * dec_outs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # should forward still take memory_lengths as a kwarg in all cases?
        tgt_emb = self.embeddings(tgt, **kwargs)

        assert tgt_emb.size(1) == self.state["input_feed"].size(1)

        attns = defaultdict(list)

        input_feeds = [self.state["input_feed"].squeeze(0)]
        dec_state = self.state["hidden"]

        for emb_t in tgt_emb.split(1):
            emb_t = emb_t.squeeze(0)
            input_feed = input_feeds[-1]
            decoder_input = torch.cat([emb_t, input_feed], 1)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)
            decoder_output, p_attn = self.attn(
                rnn_output.unsqueeze(1), memory_bank, **kwargs)
            decoder_output = decoder_output.squeeze(0)
            # p_attn used to be a tensor. Now it is a dictionary with string
            # keys and tensor values.
            for k, v in p_attn.items():
                attns[k].append(v.squeeze(0))

            decoder_output = self.dropout(decoder_output)
            input_feeds.append(decoder_output)

        dec_outs = torch.stack(input_feeds[1:])
        for k in attns:
            attns[k] = torch.stack(attns[k])

        if not isinstance(dec_state, tuple):
            dec_state = dec_state,
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)

        return dec_outs, attns
