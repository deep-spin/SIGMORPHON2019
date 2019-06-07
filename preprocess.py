#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configargparse
import glob
import sys
from functools import partial

import torch
from torchtext.data import Field

from onmt.utils.logging import init_logger, logger

import onmt.opts as opts
from onmt.inputters.dataset_base import SigmorphonDataset, lang_name


def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid', 'vocab']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, "
                             "to avoid overwriting them!\n" % path)
            sys.exit(1)


def parse_args():
    parser = configargparse.ArgumentParser(
        description='preprocess.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    check_existing_pt_files(opt)

    return opt


def get_fields(bos='<s>', eos='</s>', inflection=False,
               language=False, share_vocab=False):
    """
    language: whether to also create a field for the language of the sample
        (as for multilingual training)
    share_vocab: determines whether to create separate fields for src and
        tgt sequences or use one in common
    returns: A dictionary. The keys are minimally 'src', 'tgt', 'inflection'.
        If language is True, a 'language' key will also be present.
    """
    fields = {'src': [], 'tgt': []}

    assert not share_vocab or inflection

    if not share_vocab:
        if inflection:
            src = Field(tokenize=list, include_lengths=True)
        else:
            # total kludge here in order to get the baseline to work
            src = Field(tokenize=str.split, include_lengths=True)
        tgt = Field(init_token=bos, eos_token=eos, tokenize=list)
    else:
        src = Field(init_token=bos, eos_token=eos,
                    tokenize=list, include_lengths=True)
        tgt = src

    fields['src'].append(('src', src))
    fields['tgt'].append(('tgt', tgt))

    if inflection:
        tokenize_infl = partial(str.split, sep=';')
        infl = Field(tokenize=tokenize_infl, include_lengths=True)

        fields['inflection'] = [('inflection', infl)]

    if language:
        lang = Field(sequential=False)
        fields['language'] = [('language', lang)]

    return fields


def main():
    opt = parse_args()

    init_logger(opt.log_file)

    logger.info("Building `Fields` object...")
    fields = get_fields(
        language=opt.multilingual, share_vocab=opt.share_vocab,
        inflection=opt.inflection_field)

    # make datasets
    # allow only one path per language (this is SIGMORPHON-specific)
    train_languages = {lang_name(path): path for path in opt.train}
    opt.train = list(train_languages.values())
    train_dataset = SigmorphonDataset(
        fields, opt.train, lang_src=opt.lang_src,
        high_oversampling=opt.high_oversampling,
        low_oversampling=opt.low_oversampling)
    logger.info('Train set size: {}'.format(len(train_dataset)))

    if not opt.allow_unseen_languages:
        train_languages = {lang_name(path) for path in opt.train}
        opt.valid = [path for path in opt.valid
                     if lang_name(path) in train_languages]

    valid_dataset = SigmorphonDataset(fields, opt.valid, lang_src=opt.lang_src)
    logger.info('Validation set size: {}'.format(len(valid_dataset)))

    # build vocab
    for in_label, column_fields in fields.items():
        for name, field in column_fields:
            if field.use_vocab:
                field.build_vocab(train_dataset)

    # log some stuff
    src_vocab_size = len(fields['src'][0][1].vocab)
    tgt_vocab_size = len(fields['tgt'][0][1].vocab)
    logger.info("Vocab sizes: src {} ; tgt {}".format(
        src_vocab_size, tgt_vocab_size))
    if "inflection" in fields:
        infl_vocab_size = len(fields['inflection'][0][1].vocab)
        logger.info("Unique inflectional tags: {}".format(infl_vocab_size))
    if 'language' in fields:
        n_languages = len(fields['language'][0][1].vocab)
        print(fields['language'][0][1].vocab.itos)
        logger.info("Number of languages: {}".format(n_languages))

    train_dataset.save(opt.save_data + '.train.pt')
    valid_dataset.save(opt.save_data + '.valid.pt')
    torch.save(fields, opt.save_data + '.vocab.pt')


if __name__ == "__main__":
    main()
