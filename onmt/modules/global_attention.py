import torch
import torch.nn as nn

from onmt.modules.sparse_activations import Sparsemax
from onmt.utils.misc import aeq


def sequence_mask(lengths, max_len=None):
    """Creates a boolean mask from sequence lengths."""
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class DotScorer(nn.Module):
    def __init__(self):
        super(DotScorer, self).__init__()

    def forward(self, h_t, h_s):
        return torch.bmm(h_t, h_s.transpose(1, 2))


class GeneralScorer(nn.Module):
    def __init__(self, dim):
        super(GeneralScorer, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)

    def forward(self, h_t, h_s):
        h_t = self.linear_in(h_t)
        return torch.bmm(h_t, h_s.transpose(1, 2))


class AttentionHead(nn.Module):
    def __init__(self, score, transform):
        super(AttentionHead, self).__init__()
        self.score = score
        self.transform = transform

    @classmethod
    def from_options(cls, dim, attn_type="dot", attn_func="softmax"):
        str2score = {
            "dot": DotScorer(),
            "general": GeneralScorer(dim)
        }
        str2func = {
            "softmax": nn.Softmax(dim=-1),
            "sparsemax": Sparsemax(dim=-1)
        }
        score = str2score[attn_type]
        transform = str2func[attn_func]

        return cls(score, transform)

    def forward(self, query, memory_bank, memory_lengths=None, **kwargs):
        """
        query (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
        memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
        memory_lengths (`LongTensor`): the source context lengths `[batch]`

        returns attention distribution (tgt_len x batch x src_len)
        """
        src_batch, src_len, src_dim = memory_bank.size()
        tgt_batch, tgt_len, tgt_dim = query.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)

        align = self.score(query, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            align.masked_fill_(1 - mask.unsqueeze(1), -float('inf'))

        # it should not be necessary to view align as a 2d tensor, but
        # something is broken with sparsemax and it cannot handle a 3d tensor
        return self.transform(align.view(-1, src_len)).view_as(align)


class Attention(nn.Module):
    def __init__(self, attention_head, output_layer):
        super(Attention, self).__init__()
        self.attention_head = attention_head
        self.output_layer = output_layer

    @classmethod
    def from_options(cls, dim, attn_type="dot", attn_func="softmax"):
        attention_head = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=attn_func)
        # not gonna worry about Bahdanau attention...
        output_layer = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False), nn.Tanh()
        )
        return cls(attention_head, output_layer)

    def forward(self, query, memory_bank, memory_lengths=None, **kwargs):
        """
        query (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
        memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
        memory_lengths (`LongTensor`): the source context lengths `[batch]`

        returns attentional hidden state (tgt_len x batch x dim) and
        attention distribution (tgt_len x batch x src_len)
        """
        memory_bank = memory_bank.transpose(0, 1)

        align_vectors = self.attention_head(query, memory_bank, memory_lengths)

        # each context vector c_t is the weighted average over source states
        c = torch.bmm(align_vectors, memory_bank)

        # concatenate (this step is different in the multiheaded case)
        concat_c = torch.cat([c, query], 2)
        attn_h = self.output_layer(concat_c)

        attn_h = attn_h.transpose(0, 1).contiguous()
        align_vectors = align_vectors.transpose(0, 1).contiguous()

        return attn_h, {"lemma": align_vectors}


class TwoHeadedAttention(nn.Module):

    def __init__(self, lemma_attn, inflection_attn, output_layer):
        super(TwoHeadedAttention, self).__init__()
        self.lemma_attn = lemma_attn
        self.inflection_attn = inflection_attn
        self.output_layer = output_layer

    @classmethod
    def from_options(cls, dim, attn_type="dot", attn_func="softmax"):
        lemma_attn = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=attn_func)
        inflection_attn = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=attn_func)
        attn_output_layer = nn.Sequential(
            nn.Linear(dim * 3, dim, bias=False), nn.Tanh()
        )
        return cls(lemma_attn, inflection_attn, attn_output_layer)

    def forward(self, query, memory_bank, inflection_memory_bank,
                memory_lengths=None, inflection_lengths=None, **kwargs):
        """
        query: FloatTensor, (batch size x target len x rnn size)
        memory_bank: FloatTensor, (source len x batch size x rnn size)
        inflection_memory_bank: FloatTensor, (infl len x batch size x rnn size)
        *lengths: LongTensor, (batch_size)

        returns attentional hidden state (tgt_len x batch x dim) and
        attention distribution (tgt_len x batch x src_len)
        """
        memory_bank = memory_bank.transpose(0, 1)
        inflection_memory_bank = inflection_memory_bank.transpose(0, 1)

        lemma_align = self.lemma_attn(query, memory_bank, memory_lengths)
        infl_align = self.inflection_attn(
            query, inflection_memory_bank, inflection_lengths)
        lemma_context = torch.bmm(lemma_align, memory_bank)
        infl_context = torch.bmm(infl_align, inflection_memory_bank)
        concat_context = torch.cat([lemma_context, infl_context, query], 2)

        attn_h = self.output_layer(concat_context)

        attn_h = attn_h.transpose(0, 1).contiguous()
        lemma_align = lemma_align.transpose(0, 1).contiguous()
        infl_align = infl_align.transpose(0, 1).contiguous()
        return attn_h, {"lemma": lemma_align, "inflection": infl_align}


class GatedTwoHeadedAttention(nn.Module):

    def __init__(self, lemma_attn, inflection_attn, output_layer, gate):
        super(GatedTwoHeadedAttention, self).__init__()
        self.lemma_attn = lemma_attn
        self.inflection_attn = inflection_attn
        self.output_layer = output_layer
        self.gate = gate

    @classmethod
    def from_options(cls, dim, attn_type="dot",
                     attn_func="softmax", gate_func="softmax"):
        lemma_attn = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=attn_func)
        inflection_attn = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=attn_func)
        attn_output_layer = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False), nn.Tanh()
        )
        str2func = {
            "softmax": nn.Softmax(dim=-1), "sparsemax": Sparsemax(dim=-1)
        }
        gate_transform = str2func[gate_func]

        # try it with bias?
        gate = nn.Sequential(nn.Linear(dim * 3, 2, bias=True), gate_transform)
        return cls(lemma_attn, inflection_attn, attn_output_layer, gate)

    def forward(self, query, memory_bank, inflection_memory_bank,
                memory_lengths=None, inflection_lengths=None, **kwargs):
        """
        query: FloatTensor, (batch size x target len x rnn size)
        memory_bank: FloatTensor, (source len x batch size x rnn size)
        inflection_memory_bank: FloatTensor, (infl len x batch size x rnn size)
        *lengths: LongTensor, (batch_size)

        returns attentional hidden state (tgt_len x batch x dim) and
        attention distribution (tgt_len x batch x src_len)
        """
        batch_size, tgt_len, rnn_size = query.size()

        memory_bank = memory_bank.transpose(0, 1)
        inflection_memory_bank = inflection_memory_bank.transpose(0, 1)

        lemma_align = self.lemma_attn(query, memory_bank, memory_lengths)
        infl_align = self.inflection_attn(
            query, inflection_memory_bank, inflection_lengths)
        lemma_context = torch.bmm(lemma_align, memory_bank)
        infl_context = torch.bmm(infl_align, inflection_memory_bank)

        concat_context = torch.cat([lemma_context, infl_context, query], 2)
        concat_context = concat_context.view(batch_size * tgt_len, -1)
        gate_vec = self.gate(concat_context).view(batch_size, tgt_len, -1, 1)

        lemma_attn_h = self.output_layer(torch.cat([query, lemma_context], 2))
        infl_attn_h = self.output_layer(torch.cat([query, infl_context], 2))
        stacked_h = torch.stack([lemma_attn_h, infl_attn_h], dim=2)
        attn_h = (gate_vec * stacked_h).sum(2)

        attn_h = attn_h.transpose(0, 1).contiguous()
        lemma_align = lemma_align.transpose(0, 1).contiguous()
        infl_align = infl_align.transpose(0, 1).contiguous()
        gate = gate_vec.transpose(0, 1).squeeze(3).contiguous()

        attns = {"lemma": lemma_align, "inflection": infl_align, "gate": gate}
        return attn_h, attns


class UnsharedGatedTwoHeadedAttention(nn.Module):
    """
    This, like many of the attention mechanisms described, can probably be
    refactored away. But not until after the SIGMORPHON submission.
    """

    def __init__(self, lemma_attn, inflection_attn,
                 lemma_output_layer, inflection_output_layer, gate):
        super(UnsharedGatedTwoHeadedAttention, self).__init__()
        self.lemma_attn = lemma_attn
        self.inflection_attn = inflection_attn
        self.lemma_out = lemma_output_layer
        self.infl_out = inflection_output_layer
        self.gate = gate

    @classmethod
    def from_options(cls, dim, attn_type="dot",
                     attn_func="softmax", gate_func="softmax"):
        lemma_attn = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=attn_func)
        inflection_attn = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=attn_func)
        lemma_out = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False), nn.Tanh()
        )
        infl_out = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False), nn.Tanh()
        )
        str2func = {
            "softmax": nn.Softmax(dim=-1), "sparsemax": Sparsemax(dim=-1)
        }
        gate_transform = str2func[gate_func]

        # try it with bias?
        gate = nn.Sequential(nn.Linear(dim * 3, 2, bias=True), gate_transform)
        return cls(lemma_attn, inflection_attn, lemma_out, infl_out, gate)

    def forward(self, query, memory_bank, inflection_memory_bank,
                memory_lengths=None, inflection_lengths=None, **kwargs):
        """
        query: FloatTensor, (batch size x target len x rnn size)
        memory_bank: FloatTensor, (source len x batch size x rnn size)
        inflection_memory_bank: FloatTensor, (infl len x batch size x rnn size)
        *lengths: LongTensor, (batch_size)

        returns attentional hidden state (tgt_len x batch x dim) and
        attention distribution (tgt_len x batch x src_len)
        """
        batch_size, tgt_len, rnn_size = query.size()

        memory_bank = memory_bank.transpose(0, 1)
        inflection_memory_bank = inflection_memory_bank.transpose(0, 1)

        lemma_align = self.lemma_attn(query, memory_bank, memory_lengths)
        infl_align = self.inflection_attn(
            query, inflection_memory_bank, inflection_lengths)
        lemma_context = torch.bmm(lemma_align, memory_bank)
        infl_context = torch.bmm(infl_align, inflection_memory_bank)

        concat_context = torch.cat([lemma_context, infl_context, query], 2)
        concat_context = concat_context.view(batch_size * tgt_len, -1)
        gate_vec = self.gate(concat_context).view(batch_size, tgt_len, -1, 1)

        lemma_attn_h = self.lemma_out(torch.cat([query, lemma_context], 2))
        infl_attn_h = self.infl_out(torch.cat([query, infl_context], 2))
        stacked_h = torch.stack([lemma_attn_h, infl_attn_h], dim=2)
        attn_h = (gate_vec * stacked_h).sum(2)

        attn_h = attn_h.transpose(0, 1).contiguous()
        lemma_align = lemma_align.transpose(0, 1).contiguous()
        infl_align = infl_align.transpose(0, 1).contiguous()
        gate = gate_vec.transpose(0, 1).squeeze(3).contiguous()

        attns = {"lemma": lemma_align, "inflection": infl_align, "gate": gate}
        return attn_h, attns
