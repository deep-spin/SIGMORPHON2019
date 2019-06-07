from datetime import datetime
import sys
import time
import math
from collections import Counter
from functools import partial

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

import onmt.utils

from onmt.utils.logging import logger
from onmt.utils.misc import side_information

from onmt.modules.sparse_losses import SparsemaxLoss


# now some really ugly functions for computing metrics:


def accuracy(n_correct, n_words, **kwargs):
    return 100 * n_correct / n_words


def xent(loss, n_words, **kwargs):
    return loss / n_words


def ppl(loss, n_words, **kwargs):
    return math.exp(min(xent(loss, n_words), 100))


def wps(n_words, t, **kwargs):
    return n_words / t


def support_size(support, n_words, **kwargs):
    return support / n_words


def attn_support(attn_type, n_words, **kwargs):
    return kwargs.get(attn_type + "_attended", 0) / n_words


def gate_support(gate_value, n_words, **kwargs):
    return kwargs.get(gate_value + "_gate", 0) / n_words


def build_trainer(opt, model_opt, device_id, model, fields, optim):
    tgt_field = fields['tgt'][0][1]
    padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]

    if model_opt.loss == 'sparsemax':
        loss = SparsemaxLoss(ignore_index=padding_idx, reduction='sum')
    else:
        loss = nn.CrossEntropyLoss(ignore_index=padding_idx, reduction='sum')

    trunc_size = opt.truncated_decoder  # Badly named...

    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        tensorboard_log_dir = opt.tensorboard_log_dir

        if not opt.train_from:
            tensorboard_log_dir += datetime.now().strftime("/%b-%d_%H-%M-%S")

        writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
    else:
        writer = None

    trainer = onmt.Trainer(model, model_opt, fields, loss, loss,
                           optim, opt.save_model, trunc_size,
                           tensorboard_writer=writer,
                           report_steps=opt.report_every,
                           max_grad_norm=opt.max_grad_norm,
                           patience=opt.patience,
                           learning_rate_decay=opt.learning_rate_decay,
                           valid_metric=opt.valid_metric)
    return trainer


class Trainer(object):

    def __init__(self, model, model_opt, fields, train_loss, valid_loss, optim,
                 save_model, trunc_size=0, tensorboard_writer=None,
                 report_steps=50, max_grad_norm=5, patience=0,
                 learning_rate_decay=0.5, valid_metric="acc"):

        self.model = model
        self.model_opt = model_opt
        self.fields = fields
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.tb_writer = tensorboard_writer
        self.report_steps = report_steps
        self.save_model = save_model
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.learning_rate_decay = learning_rate_decay
        self.valid_metric = valid_metric

        # Set model in training mode.
        self.model.train()

        self.metrics = [accuracy, ppl, support_size]
        self.templates = ["acc: {:6.2f}", "ppl: {:5.2f}", "support: {:5.2f}"]
        self.metric_names = ["accuracy", "ppl", "support"]
        if self.log_sparse_attn:
            for attn_type in ["lemma", "inflection"]:
                support_func = partial(attn_support, attn_type)
                self.metrics.append(support_func)
                self.templates.append(attn_type + " attn: {:6.2f}")
                self.metric_names.append(attn_type + "_attn_support")
        if self.log_gate:
            for gate_value in ["lemma", "inflection", "both"]:
                gate_support_func = partial(gate_support, gate_value)
                self.metrics.append(gate_support_func)
                self.templates.append(gate_value + " gate: {:1.4f}")
                self.metric_names.append(gate_value + "_gate_support")
        self.metrics.append(wps)
        self.templates.append("{:3.0f} tok/s")
        self.metric_names.append("tgtper")

    @property
    def log_sparse_attn(self):
        return self.model_opt.global_attention_function == "sparsemax"

    @property
    def log_gate(self):
        return self.model_opt.inflection_gate == "sparsemax"

    def train(self, train_iter, valid_iter, epochs):
        logger.info('Start training...')
        corpus_batches = len(train_iter)
        scheduler = ReduceLROnPlateau(
            self.optim,
            mode="max" if self.valid_metric == "acc" else "min",
            factor=self.learning_rate_decay,
            patience=self.patience)
        best_acc = float("-inf")
        for i in range(1, epochs + 1):
            logger.info('Training epoch {}'.format(i))
            epoch_start = time.time()
            train_stats = self.train_epoch(train_iter)
            valid_stats = self.validate(valid_iter)
            self._report_step(
                corpus_batches * i, train_stats, "train", epoch_start)
            self._report_step(
                corpus_batches * i, valid_stats, "valid", epoch_start)
            metric = accuracy(**valid_stats) if self.valid_metric == "acc" \
                else ppl(**valid_stats)  # not actually the loss. Rename?
            if metric > best_acc:
                # this would require refactoring if using loss instead
                self.save(i)
                best_acc = metric
            scheduler.step(metric)
        if self.tb_writer is not None:
            self.tb_writer.close()

    def log_stats(self, stats, step, num_steps, epoch_start, report_start):
        report = ["Step {}/{}".format(step, num_steps)]
        report.append("lr: {:7.5f}".format(self.learning_rate))
        t = time.time() - report_start

        results = [metric(**stats, t=t + 1e-5) for metric in self.metrics]
        report.extend([t.format(s) for t, s in zip(self.templates, results)])
        report.append("{:6.0f} sec".format(time.time() - epoch_start))
        logger.info("; ".join(report))
        sys.stdout.flush()

    def train_epoch(self, train_iter):
        train_size = len(train_iter)
        epoch_stats = Counter()
        report_stats = Counter()

        epoch_start = time.time()
        report_start = epoch_start

        for step, batch in enumerate(train_iter, 1):
            batch_stats = self.train_batch(batch)
            epoch_stats.update(batch_stats)
            report_stats.update(batch_stats)

            if step % self.report_steps == 0:
                self.log_stats(
                    report_stats, step, train_size, epoch_start, report_start)
                if self.tb_writer is not None:
                    self._log_tensorboard(
                        report_stats, step, "progress", epoch_start)
                report_stats.clear()
                report_start = time.time()
        return epoch_stats

    def validate(self, valid_iter):
        with torch.no_grad():
            self.model.eval()

            stats = Counter()

            for batch in valid_iter:
                src, lengths = batch.src
                tgt = batch.tgt[0] if isinstance(batch.tgt, tuple) \
                    else batch.tgt
                side_info = side_information(batch)

                outputs, attns = self.model(
                    src, tgt, lengths=lengths, **side_info)
                logits = self.model.generator(outputs, **side_info)
                logits = logits.view(-1, logits.size(-1))
                gold = tgt[1:].view(-1)

                loss = self.valid_loss(logits, gold)
                if isinstance(loss, tuple):
                    loss, p_star = loss
                else:
                    p_star = None

                pred = logits.max(1)[1]
                non_pad = gold.ne(self.valid_loss.ignore_index)
                n_correct = pred.eq(gold).masked_select(non_pad).sum().item()
                n_words = non_pad.sum().item()
                batch_loss = loss.clone().item()
                if self.log_sparse_attn:
                    for k, v in attns.items():
                        bottled_attn = v.view(v.size(0) * v.size(1), -1)
                        attn_supp = bottled_attn.gt(0).sum(dim=1)
                        n_attended = attn_supp.masked_select(non_pad)
                        stats[k + "_attended"] += n_attended.sum().item()
                if self.log_gate:
                    seq_len, batch_size, classes = attns["gate"].size()
                    bottled_gate = attns["gate"].view(seq_len * batch_size, -1)
                    gate_support = bottled_gate.gt(0)
                    lemma_gate = gate_support[:, 0].masked_select(non_pad)
                    infl_gate = gate_support[:, 1].masked_select(non_pad)
                    both_gate = lemma_gate & infl_gate
                    stats["lemma_gate"] += lemma_gate.sum().item()
                    stats["inflection_gate"] += infl_gate.sum().item()
                    stats["both_gate"] += both_gate.sum().item()
                if p_star is not None:
                    support = p_star.gt(0).sum(dim=1)
                    support = support.masked_select(non_pad).sum().item()
                    stats["support"] += support
                else:
                    stats["support"] += n_words * logits.size(-1)

                stats["loss"] += batch_loss
                stats["n_words"] += n_words
                stats["n_correct"] += n_correct

            self.model.train()

            return stats

    def train_batch(self, batch):
        self.model.zero_grad()  # right thing to do?
        src, lengths = batch.src
        tgt_outer = batch.tgt[0] if isinstance(batch.tgt, tuple) else batch.tgt

        side_info = side_information(batch)

        target_size = tgt_outer.size(0)

        trunc_size = self.trunc_size if self.trunc_size else target_size
        batch_stats = Counter()

        for j in range(0, target_size - 1, trunc_size):
            # 1. Create truncated target.
            tgt = tgt_outer[j: j + trunc_size]

            self.model.zero_grad()
            outputs, attns = self.model(src, tgt, lengths=lengths, **side_info)

            logits = self.model.generator(outputs, **side_info)
            logits = logits.view(-1, logits.size(-1))
            gold = tgt[1:].view(-1)

            loss = self.train_loss(logits, gold)
            if isinstance(loss, tuple):
                loss, p_star = loss
            else:
                p_star = None
            loss.div(batch.batch_size).backward()

            # this stuff should be refactored
            pred = logits.max(1)[1]
            non_pad = gold.ne(self.train_loss.ignore_index)
            n_correct = pred.eq(gold).masked_select(non_pad).sum().item()
            n_words = non_pad.sum().item()
            batch_loss = loss.clone().item()
            if self.log_sparse_attn:
                for k, v in attns.items():
                    bottled_attn = v.view(v.size(0) * v.size(1), -1)
                    attn_support = bottled_attn.gt(0).sum(dim=1)
                    n_attended = attn_support.masked_select(non_pad)
                    batch_stats[k + "_attended"] += n_attended.sum().item()
            if self.log_gate:
                seq_len, batch_size, classes = attns["gate"].size()
                bottled_gate = attns["gate"].view(seq_len * batch_size, -1)
                gate_support = bottled_gate.gt(0)
                lemma_gate = gate_support[:, 0].masked_select(non_pad)
                infl_gate = gate_support[:, 1].masked_select(non_pad)
                both_gate = lemma_gate & infl_gate
                batch_stats["lemma_gate"] += lemma_gate.sum().item()
                batch_stats["inflection_gate"] += infl_gate.sum().item()
                batch_stats["both_gate"] += both_gate.sum().item()
            if p_star is not None:
                support = p_star.gt(0).sum(dim=1)
                support = support.masked_select(non_pad).sum().item()
                batch_stats["support"] += support
            else:
                batch_stats["support"] += n_words * logits.size(-1)

            batch_stats["loss"] += batch_loss
            batch_stats["n_words"] += n_words
            batch_stats["n_correct"] += n_correct

            for group in self.optim.param_groups:
                if self.max_grad_norm:
                    clip_grad_norm_(group['params'], self.max_grad_norm)
            self.optim.step()

            # If truncated, don't backprop fully.
            if self.model.decoder.state is not None:
                self.model.decoder.detach_state()
        return batch_stats

    def _report_step(self, step, stats, label, epoch_start):
        logger.info('{} perplexity: {:.4f}'.format(label, ppl(**stats)))
        logger.info('{} accuracy: {:.4f}'.format(label, accuracy(**stats)))
        if self.log_sparse_attn:
            for attn_type in ["lemma", "inflection"]:
                support_func = partial(attn_support, attn_type)
                logger.info("{} {} attn: {:6.2f}".format(
                    label, attn_type, support_func(**stats)))
        if self.log_gate:
            for gate_value in ["lemma", "inflection", "both"]:
                gate_func = partial(gate_support, gate_value)
                logger.info("{} {} gate: {:6.2f}".format(
                    label, gate_value, gate_func(**stats)))
        logger.info("{} support: {:6.2f}".format(label, support_size(**stats)))

        if self.tb_writer is not None:
            self._log_tensorboard(stats, step, label, epoch_start)

    @property
    def learning_rate(self):
        return self.optim.param_groups[0]["lr"]

    def save(self, epoch):
        checkpoint = {
            'model': self.model.state_dict(),
            'vocab': self.fields,
            'opt': self.model_opt,
            'optim': self.optim,
        }

        out_path = "{}_epoch_{}.pt".format(self.save_model, epoch)
        logger.info("Saving checkpoint {}".format(out_path))
        torch.save(checkpoint, out_path)

    def _log_tensorboard(self, stats, step, prefix, start_time):
        t = time.time() - start_time
        for name, metric in zip(self.metric_names, self.metrics):
            name = "/".join([prefix, name])
            self.tb_writer.add_scalar(name, metric(**stats, t=t), step)

        self.tb_writer.add_scalar(prefix + "/lr", self.learning_rate, step)
