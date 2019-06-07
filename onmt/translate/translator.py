import configargparse

import torch
from torchtext.data.iterator import Iterator

from onmt.utils.misc import tile, side_information

import onmt.model_builder
from onmt.translate.beam import Beam
from onmt.inputters.dataset_base import SigmorphonDataset
import onmt.opts as opts
import onmt.decoders.ensemble


def build_translator(opt):
    out_file = open(opt.output, 'w+', encoding='utf-8')

    dummy_parser = configargparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    load_test_model = onmt.decoders.ensemble.load_test_model \
        if len(opt.models) > 1 else onmt.model_builder.load_test_model
    fields, model, _ = load_test_model(opt, dummy_opt.__dict__)

    return Translator(model, fields, opt, out_file=out_file)


class Translator(object):

    def __init__(self, model, fields, opt, out_file=None):
        self.model = model
        self.fields = fields
        self.cuda = opt.gpu > -1

        self.n_best = opt.n_best
        self.max_length = opt.max_length
        self.beam_size = opt.beam_size
        self.verbose = opt.verbose
        self.attn_path = opt.attn_path
        self.probs_path = opt.probs_path
        self.attns = []
        self.probs = []

        self.out_file = out_file
        self.dump_beam = opt.dump_beam
        if opt.dump_beam != "":
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": []}
        else:
            self.beam_accum = None

    def _tgt_str(self, pred):
        tgt_field = self.fields["tgt"][0][1]
        vocab = tgt_field.vocab
        tokens = []
        for tok in pred:
            tokens.append(vocab.itos[tok])
            if tokens[-1] == tgt_field.eos_token:
                tokens = tokens[:-1]
                break
        return tokens

    def itos(self, sequence, field_name):
        vocab = self.fields[field_name][0][1].vocab
        return [vocab.itos[i] for i in sequence]

    def log(self, src, inflection, pred_sents, pred_scores,
            gold_sent, gold_score, i):
        src_toks = ' '.join(self.itos(src, "src"))
        if "inflection" in self.fields:
            infl_vocab = self.fields["inflection"][0][1].vocab
            infl_tags = " ".join(infl_vocab.itos[j] for j in inflection)
            print('\nSENT {}: {} {}'.format(i, infl_tags, src_toks))
        else:
            print('\nSENT {}: {}'.format(i, src_toks))

        best_pred = pred_sents[0]
        best_score = pred_scores[0]
        pred_sent = ' '.join(best_pred)
        print('PRED {}: {}'.format(i, pred_sent))
        print("PRED SCORE: {:.4f}".format(best_score))

        if gold_sent is not None:
            tgt_sent = ' '.join(gold_sent)
            print('GOLD {}: {}'.format(i, tgt_sent))
            print("GOLD SCORE: {:.4f}".format(gold_score))
        if len(pred_sents) > 1:
            print('\nBEST HYP:')
            for score, sent in zip(pred_scores, pred_sents):
                print("[{:.4f}] {}".format(score, sent))

    def translate(self, corpora):
        data = SigmorphonDataset(self.fields.copy(), corpora)

        cur_device = "cuda" if self.cuda else "cpu"

        data_iter = Iterator(
            data, batch_size=1, sort_within_batch=True,
            device=cur_device, repeat=False, shuffle=False)

        # Statistics
        # counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        for i, batch in enumerate(data_iter, 1):
            with torch.no_grad():
                batch_results = self.translate_batch(batch)

            preds = batch_results["preds"]
            pred_score = batch_results["scores"]
            gold_score = batch_results["gold_score"]

            src, src_len = batch.src
            if hasattr(batch, 'tgt'):
                if isinstance(batch.tgt, tuple):
                    tgt = batch.tgt[0]
                else:
                    tgt = batch.tgt
            else:
                tgt = None
            if hasattr(batch, "inflection"):
                inflection = batch.inflection[0]
            else:
                inflection = None

            pred_sents = [self._tgt_str(preds[n]) for n in range(self.n_best)]
            gold_sent = self._tgt_str(tgt[1:]) if tgt is not None else None

            pred_score_total += pred_score[0]
            pred_words_total += len(pred_sents[0])

            if tgt is not None:
                gold_score_total += gold_score
                gold_words_total += len(gold_sent) + 1

            # don't print n_best to file
            self.out_file.write(" ".join(pred_sents[0]) + '\n')
            self.out_file.flush()

            if self.verbose:
                self.log(
                    src, inflection, pred_sents, pred_score,
                    gold_sent, gold_score, i)

        if self.dump_beam:
            import json
            json.dump(self.beam_accum, open(self.dump_beam, 'w'))
        if self.attn_path is not None:
            torch.save(self.attns, self.attn_path)
        if self.probs_path is not None:
            torch.save(self.probs, self.probs_path)

    def translate_batch(self, batch):
        beam_size = self.beam_size
        tgt_field = self.fields['tgt'][0][1]
        vocab = tgt_field.vocab

        pad = vocab.stoi[tgt_field.pad_token]
        eos = vocab.stoi[tgt_field.eos_token]
        bos = vocab.stoi[tgt_field.init_token]
        b = Beam(beam_size, n_best=self.n_best, cuda=self.cuda,
                 pad=pad, eos=eos, bos=bos)

        src, src_lengths = batch.src
        # why doesn't this contain inflection source lengths when ensembling?
        side_info = side_information(batch)

        encoder_out = self.model.encode(src, lengths=src_lengths, **side_info)
        enc_states = encoder_out["enc_state"]
        memory_bank = encoder_out["memory_bank"]
        infl_memory_bank = encoder_out.get("inflection_memory_bank", None)

        self.model.init_decoder_state(enc_states)

        results = dict()

        if "tgt" in batch.__dict__:
            results["gold_score"] = self._score_target(
                batch, memory_bank, src_lengths,
                inflection_memory_bank=infl_memory_bank, **side_info)
            self.model.init_decoder_state(enc_states)
        else:
            results["gold_score"] = 0

        # (2) Repeat src objects `beam_size` times.
        self.model.map_decoder_state(
            lambda state, dim: tile(state, beam_size, dim=dim))

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, beam_size, dim=1) for x in memory_bank)
        else:
            memory_bank = tile(memory_bank, beam_size, dim=1)
        memory_lengths = tile(src_lengths, beam_size)

        if infl_memory_bank is not None:
            if isinstance(infl_memory_bank, tuple):
                infl_memory_bank = tuple(tile(x, beam_size, dim=1)
                                         for x in infl_memory_bank)
            else:
                infl_memory_bank = tile(infl_memory_bank, beam_size, dim=1)
            tiled_infl_len = tile(side_info["inflection_lengths"], beam_size)
            side_info["inflection_lengths"] = tiled_infl_len

        if "language" in side_info:
            side_info["language"] = tile(side_info["language"], beam_size)

        for i in range(self.max_length):
            if b.done():
                break

            inp = b.current_state.unsqueeze(0)

            # the decoder expects an input of tgt_len x batch
            dec_out, dec_attn = self.model.decode(
                inp, memory_bank, memory_lengths=memory_lengths,
                inflection_memory_bank=infl_memory_bank, **side_info
            )
            attn = dec_attn["lemma"].squeeze(0)
            out = self.model.generator(
                dec_out.squeeze(0), transform=True, **side_info
            )

            # b.advance will take attn (beam size x src length)
            b.advance(out, dec_attn)
            select_indices = b.current_origin

            self.model.map_decoder_state(
                lambda state, dim: state.index_select(dim, select_indices))

        scores, ks = b.sort_finished()
        hyps, attn, out_probs = [], [], []
        for i, (times, k) in enumerate(ks[:self.n_best]):
            hyp, att, out_p = b.get_hyp(times, k)
            hyps.append(hyp)
            attn.append(att)
            out_probs.append(out_p)

        results["preds"] = hyps
        results["scores"] = scores
        results["attn"] = attn

        if self.beam_accum is not None:
            parent_ids = [t.tolist() for t in b.prev_ks]
            self.beam_accum["beam_parent_ids"].append(parent_ids)
            scores = [["%4f" % s for s in t.tolist()]
                      for t in b.all_scores][1:]
            self.beam_accum["scores"].append(scores)
            pred_ids = [[vocab.itos[i] for i in t.tolist()]
                        for t in b.next_ys][1:]
            self.beam_accum["predicted_ids"].append(pred_ids)

        if self.attn_path is not None:
            save_attn = {k: v.cpu() for k, v in attn[0].items()}
            src_seq = self.itos(src, "src")
            pred_seq = self.itos(hyps[0], "tgt")
            attn_dict = {"src": src_seq, "pred": pred_seq, "attn": save_attn}
            if "inflection" in save_attn:
                inflection_seq = self.itos(batch.inflection[0], "inflection")
                attn_dict["inflection"] = inflection_seq
            self.attns.append(attn_dict)

        if self.probs_path is not None:
            save_probs = out_probs[0].cpu()
            self.probs.append(save_probs)

        return results

    def _score_target(self, batch, memory_bank, src_lengths, **kwargs):
        tgt_in = batch.tgt[:-1]

        dec_out, _ = self.model.decode(
            tgt_in, memory_bank, memory_lengths=src_lengths, **kwargs
        )
        log_probs = self.model.generator(
            dec_out.squeeze(0), transform=True, **kwargs
        )

        tgt_field = self.fields["tgt"][0][1]
        tgt_pad = tgt_field.vocab.stoi[tgt_field.pad_token]

        log_probs[:, :, tgt_pad] = 0
        gold = batch.tgt[1:].unsqueeze(2)
        gold_scores = log_probs.gather(2, gold)
        return gold_scores.sum().item()
