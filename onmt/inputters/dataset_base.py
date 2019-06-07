from os.path import basename

from itertools import chain
import re

import torch
from torchtext.data import Example, Dataset


def lang_name(path):
    filename = basename(path)
    filename = re.sub(r"-(un)?covered", "", filename)
    match = re.match(r'.+(?=-(train|dev|test|covered|uncovered))', filename)
    if match:
        return match.group(0)
    else:
        return None


def data_setting(path):
    filename = basename(path)
    return filename.split("-")[-1]


def has_tgt(path):
    with open(path) as f:
        return len(f.readline().strip().split("\t")) == 3


class SigmorphonDataset(Dataset):
    @staticmethod
    def sort_key(ex):
        if hasattr(ex, "tgt"):
            return len(ex.src), len(ex.tgt)
        return len(ex.src)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, _d):
        self.__dict__.update(_d)

    def __reduce_ex__(self, proto):
        # This is a hack. Something is broken with torch pickle.
        return super(SigmorphonDataset, self).__reduce_ex__()

    def __init__(self, fields, paths, filter_pred=None, lang_src=False,
                 high_oversampling=1, low_oversampling=1):
        if isinstance(paths, str):
            paths = [paths]
        examples = []
        for path in paths:
            with open(path) as f:
                language = lang_name(path) if 'language' in fields else None
                setting = data_setting(path)

                for line in f:
                    ex_dict = dict()
                    if language is not None:
                        ex_dict["language"] = language
                    line_fields = line.strip().split('\t')
                    if len(line_fields) == 3:
                        src, tgt, inflection = line_fields
                        ex_dict['tgt'] = tgt
                    else:
                        src, inflection = line_fields
                        fields.pop("tgt", None)
                    if "inflection" in fields:
                        ex_dict["src"] = src
                        ex_dict["inflection"] = inflection
                    else:
                        respaced_inflection = " ".join(inflection.split(";"))
                        respaced_src = " ".join(
                            [c if c != " " else "<space>" for c in src])
                        src_seq = []
                        if language is not None and lang_src:
                            src_seq.append(language)
                        src_seq.extend([respaced_inflection, respaced_src])
                        ex_dict["src"] = " ".join(src_seq)
                    ex = Example.fromdict(ex_dict, fields)
                    if setting == "low":
                        examples.extend((ex for i in range(low_oversampling)))
                    else:
                        examples.extend((ex for i in range(high_oversampling)))
        fields = dict(chain.from_iterable(fields.values()))
        super(SigmorphonDataset, self).__init__(examples, fields, filter_pred)

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)
