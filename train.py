#!/usr/bin/env python

import configargparse

import onmt.opts as opts

import os
import random
from itertools import chain

import torch
import torchtext
# from torchtext.data.iterator import Iterator

from onmt.model_builder import build_model
from onmt.trainer import build_trainer
from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import use_gpu


class OrderedIterator(torchtext.data.Iterator):

    def create_batches(self):
        if self.train:
            def _pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = _pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return n_params, enc, dec


def training_opt_postprocessing(opt, device_id):
    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc_layers = opt.layers
        opt.dec_layers = opt.layers

    if opt.rnn_size != -1:
        opt.enc_rnn_size = opt.rnn_size
        opt.dec_rnn_size = opt.rnn_size

    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(opt.seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        if opt.seed > 0:
            # These ensure same initialization in multi gpu mode
            torch.cuda.manual_seed(opt.seed)

    return opt


def main(opt):
    if opt.gpuid:
        raise AssertionError("gpuid is deprecated \
              see world_size and gpu_ranks")

    assert opt.world_size <= 1, "you don't need multi-gpu for morphology"

    device_id = 0 if len(opt.gpu_ranks) == 1 else -1

    opt = training_opt_postprocessing(opt, device_id)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)

        # Load default opts values then overwrite it with opts from
        # the checkpoint. It's useful in order to re-train a model
        # after adding a new option (not set in checkpoint)
        dummy_parser = configargparse.ArgumentParser()
        opts.model_opts(dummy_parser)
        default_opt = dummy_parser.parse_known_args([])[0]

        model_opt = default_opt
        model_opt.__dict__.update(checkpoint['opt'].__dict__)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = checkpoint['vocab']
    else:
        checkpoint = None
        model_opt = opt
        fields = torch.load(opt.data + '.vocab.pt')

    for key, values in fields.items():
        for name, f in values:
            if f.use_vocab:
                logger.info(' * %s vocab size = %d' % (name, len(f.vocab)))

    # Build model.
    logger.info('Building model...')
    model = build_model(model_opt, fields, use_gpu(opt), checkpoint)
    logger.info(model)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    params = model.parameters()
    optim_args = {"lr": opt.learning_rate}
    if opt.optim == "adam":
        # no need to mess with the default betas
        optim_args["eps"] = 1e-9
    elif opt.optim == "adagrad":
        optim_args["initial_accumulator_value"] = opt.adagrad_accumulator_init
    optim = getattr(torch.optim, opt.optim.title())(params, **optim_args)
    print(optim)

    trainer = build_trainer(opt, model_opt, device_id, model, fields, optim)

    # this line is kind of a temporary kludge because different objects expect
    # fields to have a different structure
    dataset_fields = dict(chain.from_iterable(fields.values()))

    device = "cuda" if opt.gpu_ranks else "cpu"

    train_dataset = torch.load(opt.data + '.train.pt')
    train_dataset.fields = dataset_fields
    train_iter = OrderedIterator(
        train_dataset, opt.batch_size, sort_within_batch=True,
        device=device, repeat=False, shuffle=not opt.no_shuffle)

    valid_dataset = torch.load(opt.data + '.valid.pt')
    valid_dataset.fields = dataset_fields
    valid_iter = OrderedIterator(
        valid_dataset, opt.valid_batch_size, train=False,
        sort_within_batch=True, device=device)

    logger.info('Starting training on {}'.format(device))
    trainer.train(train_iter, valid_iter, opt.epochs)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description='train.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)
