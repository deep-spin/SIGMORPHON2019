from collections import defaultdict
import torch


class Beam(object):
    """
    size (int): beam size
    """

    def __init__(self, size, pad, bos, eos, n_best=1, cuda=False):

        self.size = size  # beam size
        device = 'cuda' if cuda else 'cpu'

        self.all_scores = [torch.zeros(size, device=device)]

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), pad, device=device,
                        dtype=torch.long)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        self.attn = defaultdict(list)
        self.out_probs = []

        self.finished = []
        self.n_best = n_best

    @property
    def current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    @property
    def current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def current_score(self):
        return self.all_scores[-1]

    def advance(self, word_probs, attn_out):
        """
        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- dictionary of attentions at the last step
        """
        # Sum the previous scores.
        # it should be possible to remove the if/else, but doing so currently
        # results in duplicate full-probability hypotheses
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + self.current_score.unsqueeze(1)
            # Don't let EOS have children.
            beam_scores.masked_fill_(
                self.current_state.eq(self._eos).unsqueeze(1), -1e20)
        else:
            beam_scores = word_probs[0]
        beam_scores = beam_scores.view(-1)
        best_scores, best_id = beam_scores.topk(self.size)

        self.all_scores.append(best_scores)

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        num_words = word_probs.size(1)  # vocab size
        prev_k = best_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_id - prev_k * num_words))
        for k, v in attn_out.items():
            step_attn = v.squeeze(0).index_select(0, prev_k)
            self.attn[k].append(step_attn)
        self.out_probs.append(torch.exp(word_probs))

        for i in range(self.current_state.size(0)):
            if self.current_state[i] == self._eos:
                s = self.current_score[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.current_state[0] == self._eos:
            self.eos_top = True

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self):
        if self.n_best > 1:
            i = 0
            while len(self.finished) < self.n_best:
                s = self.current_score[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(reverse=True)
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        hyp = []
        attn = defaultdict(list)
        out_probs = []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            for label, label_attn in self.attn.items():
                attn[label].append(label_attn[j][k])
            out_probs.append(self.out_probs[j][k])
            k = self.prev_ks[j][k]
        attn = {label: torch.stack(label_attn[::-1])
                for label, label_attn in attn.items()}
        out_probs = torch.stack(out_probs)
        return hyp[::-1], attn, out_probs
