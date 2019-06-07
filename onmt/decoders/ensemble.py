"""
Ensemble decoding.

Decodes using multiple models simultaneously,
combining their prediction distributions by averaging.
All models in the ensemble must share a target vocabulary.
"""
from collections import defaultdict
from itertools import repeat

import torch
import torch.nn as nn

from onmt.models import NMTModel
import onmt.model_builder


def combine_attns(attns):
    result = {}
    for key in attns[0].keys():
        result[key] = torch.stack([attn[key] for attn in attns]).mean(0)
    return result


class EnsembleDecoderOutput(object):
    """ Wrapper around multiple decoder final hidden states """
    def __init__(self, model_dec_outs):
        self.model_dec_outs = tuple(model_dec_outs)

    def squeeze(self, dim=None):
        """
        Delegate squeeze to avoid modifying
        :obj:`Translator.translate_batch()`
        """
        return EnsembleDecoderOutput([
            x.squeeze(dim) for x in self.model_dec_outs])

    def __getitem__(self, index):
        return self.model_dec_outs[index]


class EnsembleGenerator(nn.Module):
    def __init__(self, model_generators, raw_probs=False):
        super(EnsembleGenerator, self).__init__()
        self.model_generators = nn.ModuleList(model_generators)
        self._raw_probs = raw_probs

    def forward(self, hidden, **kwargs):
        """
        Compute a distribution over the target dictionary
        by averaging distributions from models in the ensemble.
        All models in the ensemble must share a target vocabulary.
        """
        distributions = torch.stack(
            [mg(h, **kwargs) for h, mg in zip(hidden, self.model_generators)]
        )
        if self._raw_probs:
            return torch.log(torch.exp(distributions).mean(0))
        else:
            return distributions.mean(0)


class EnsembleModel(NMTModel):
    def __init__(self, models, raw_probs=False):
        generator = EnsembleGenerator(
            [model.generator for model in models], raw_probs)
        super(EnsembleModel, self).__init__(None, None, generator)
        self.models = nn.ModuleList(models)

    def encode(self, src, **kwargs):
        results = defaultdict(list)
        for model in self.models:
            for k, v in model.encode(src, **kwargs).items():
                results[k].append(v)
        return {k: tuple(v) for k, v in results.items()}

    def decode(self, tgt, memory_bank, inflection_memory_bank=None, **kwargs):
        if inflection_memory_bank is None:
            inflection_memory_bank = repeat(None)
        decoder_inputs = zip(memory_bank, inflection_memory_bank, self.models)
        dec_outs, attns = zip(*[
            model.decode(tgt, mb, inflection_memory_bank=infl_mb, **kwargs)
            for mb, infl_mb, model in decoder_inputs])
        mean_attns = combine_attns(attns)
        return EnsembleDecoderOutput(dec_outs), mean_attns

    def init_decoder_state(self, encoder_state):
        for model, hidden in zip(self.models, encoder_state):
            model.init_decoder_state(hidden)

    def map_decoder_state(self, fn):
        for model in self.models:
            model.map_decoder_state(fn)


def load_test_model(opt, dummy_opt):
    """ Read in multiple models for ensemble """
    shared_fields = None
    shared_model_opt = None
    models = []
    for model_path in opt.models:
        fields, model, model_opt = onmt.model_builder.load_test_model(
            opt, dummy_opt, model_path)
        if shared_fields is None:
            shared_fields = fields
        else:
            for key, field_list in fields.items():
                field = field_list[0][1]
                shared_field = shared_fields[key][0][1]
                if field is not None and 'vocab' in field.__dict__:
                    assert field.vocab.stoi == shared_field.vocab.stoi, \
                        'Ensemble models must use the same preprocessed data'
        models.append(model)
        if shared_model_opt is None:
            shared_model_opt = model_opt
    ensemble_model = EnsembleModel(models, opt.avg_raw_probs)
    return shared_fields, ensemble_model, shared_model_opt
