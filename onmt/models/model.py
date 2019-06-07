import torch.nn as nn


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, generator):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src, tgt, lengths, **kwargs):
        """
        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
        Returns:
            * decoder output `[tgt_len x batch x hidden]`
            * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        encoder_out = self.encode(src, lengths=lengths, **kwargs)
        self.init_decoder_state(encoder_out["enc_state"])
        dec_out, attns = self.decoder(
            tgt, encoder_out["memory_bank"], memory_lengths=lengths, **kwargs)

        return dec_out, attns

    def encode(self, src, lengths, **kwargs):
        result = dict()
        enc_state, memory_bank = self.encoder(src, lengths=lengths, **kwargs)
        result["enc_state"] = enc_state
        result["memory_bank"] = memory_bank
        return result

    def decode(self, tgt, memory_bank, **kwargs):
        return self.decoder(tgt, memory_bank, **kwargs)

    def init_decoder_state(self, encoder_state):
        self.decoder.init_state(encoder_state)

    def map_decoder_state(self, fn):
        self.decoder.map_state(fn)


class InflectionAttentionModel(NMTModel):

    def __init__(self, encoder, inflection_encoder, decoder, generator):
        super(InflectionAttentionModel, self).__init__(
            encoder, decoder, generator)
        self.inflection_encoder = inflection_encoder

    def forward(self, src, tgt, lengths,
                inflection, inflection_lengths, **kwargs):

        tgt = tgt[:-1]  # exclude last target from inputs

        encoder_out = self.encode(src, lengths, inflection, **kwargs)

        self.init_decoder_state(encoder_out["enc_state"])

        dec_out, attns = self.decode(
            tgt, encoder_out["memory_bank"], memory_lengths=lengths,
            inflection_memory_bank=encoder_out["inflection_memory_bank"],
            inflection_lengths=inflection_lengths, **kwargs)

        return dec_out, attns

    def encode(self, src, lengths, inflection, **kwargs):
        result = dict()
        enc_state, memory_bank = self.encoder(src, lengths=lengths, **kwargs)
        result["enc_state"] = enc_state
        result["memory_bank"] = memory_bank
        inflection_memory_bank = self.inflection_encoder(inflection, **kwargs)
        result["inflection_memory_bank"] = inflection_memory_bank
        return result

    def decode(self, tgt, memory_bank, memory_lengths, inflection_memory_bank,
               inflection_lengths, **kwargs):
        dec_out, attns = self.decoder(
            tgt, memory_bank, memory_lengths=memory_lengths,
            inflection_memory_bank=inflection_memory_bank,
            inflection_lengths=inflection_lengths, **kwargs)
        return dec_out, attns
