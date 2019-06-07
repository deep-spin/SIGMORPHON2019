#!/usr/bin/env python

import argparse
from itertools import compress
from collections import Counter
from os.path import basename, splitext
import torch
import pandas as pd
from plotnine import *


def language_pair(path):
    return tuple(splitext(basename(path))[0].split("--"))


def index(seq, value):
    try:
        return seq.index(value)
    except ValueError:
        return None


def id_morpheme(plot, tag):
    inflection = plot["inflection"]
    inflection_attn = plot["attn"]["inflection"]
    tag_index = index(inflection, tag)
    if tag_index is not None:
        mask = inflection_attn[:, tag_index] > 0
        print(''.join(compress(plot["pred"], mask)),  ''.join(plot["src"]))


def avg_attended(plots, attn_field):
    src_attended, tgt_generated = 0, 0
    for plot in plots:
        # each 'plot' is a dict
        matrix = plot["attn"][attn_field]
        src_attended += matrix.gt(0).sum().item()
        tgt_generated += matrix.size(0)
    return src_attended / tgt_generated


def gate_avg_attended(plots, attn_field):
    src_attended, tgt_generated = 0, 0
    for plot in plots:
        gate_weights = plot["attn"]["gate"]
        lemma_weights = gate_weights[:, 0].unsqueeze(1)
        infl_weights = gate_weights[:, 1].unsqueeze(1)

        if attn_field == "total":
            lemma_plot = plot["attn"]["lemma"] * lemma_weights
            infl_plot = plot["attn"]["inflection"] * infl_weights
            matrix = torch.cat([lemma_plot, infl_plot], dim=1)
            src_attended += matrix.gt(0).sum().item()
            tgt_generated += matrix.size(0)
        else:
            weights = lemma_weights if attn_field == "lemma" else infl_weights
            used_ix = weights.gt(0)
            matrix = plot["attn"][attn_field]
            src_attended += matrix.gt(0).masked_select(used_ix).sum().item()
            tgt_generated += used_ix.sum().item()
            
    return src_attended / tgt_generated


def tag_cooccurrences(plots):
    counts = Counter()
    for plot in plots:
        matrix = plot["attn"]["inflection"]
        tags = plot["inflection"]
        for mask in matrix.gt(0).split(1):
            mask = mask.squeeze(0)
            attended = tuple(compress(tags, mask))  # tuple works?
            counts[attended] += 1  # but do I just want the counts?
            # another thing to ask: consecutive indices with the tags active?
    return counts


def load_data(paths):
    return {language_pair(path): torch.load(path) for path in paths}


def attn_frame(double_plots=dict(), gate_plots=dict()):
    tree = dict()
    tree['turkic'] = ['azeri', "bashkir", 'crimean-tatar', 'kazakh', 'khakas',
                      'tatar', "turkish", 'turkmen', "uzbek"]
    tree['afro_asiatic'] = ["arabic", 'classical-syriac', "hebrew", 'maltese']
    tree['nw_caucasian'] = ["adyghe", "kabardian"]
    tree['albanian'] = ["albanian"]
    tree['armenian'] = ["armenian"]
    tree['basque'] = ["basque"]
    tree['romance'] = ["asturian", "french", 'friulian', "italian", 'ladin',
                       "latin", "neapolitan", "occitan", "portuguese",
                       "romanian", "spanish"]
    tree['slavic'] = ['belarusian', 'bulgarian', 'czech', "kashubian",
                      'old-church-slavonic', "polish", "russian", 'slovak',
                      'slovene']
    tree['baltic'] = ["latvian", "lithuanian"]
    tree['germanic'] = ["danish", "dutch", "english", "german",
                        'middle-high-german', 'middle-low-german',
                        'north-frisian', "old-english", "old-saxon",
                        'west-frisian', 'yiddish']
    tree['celtic'] = ["breton", "cornish", "irish", "old-irish",
                      "scottish-gaelic", "welsh"]
    tree['uralic'] = ["estonian", "finnish", "hungarian", 'ingrian',
                      'karelian', 'livonian', 'votic']
    tree['greek'] = ["greek"]
    tree['indo_iranian'] = ["bengali", "hindi", "kurmanji", "pashto",
                            "persian", "sanskrit", "sorani", "urdu"]
    tree['dravidian'] = ["kannada", "telugu"]
    tree['murrinhpatha'] = ["murrinhpatha"]
    tree['niger_congo'] = ["swahili", "zulu"]
    tree['quechua'] = ["quechua"]
    family_lookup = dict()
    for family, languages in tree.items():
        for language in languages:
            family_lookup[language] = family

    langs = sorted(double_plots)
    high, low = zip(*langs)
    attended = dict()
    for field in ["lemma", "inflection"]:
        attended["double_" + field] = [avg_attended(double_plots[k], field)
                                       for k in sorted(double_plots)]
    for field in ["lemma", "inflection", "total"]:
        attended["gate_" + field] = [gate_avg_attended(gate_plots[k], field)
                                     for k in sorted(gate_plots)]

    '''
        
    if not gate:
        for field in ["lemma", "inflection"]:
            attended[field] = [avg_attended(plots[k], field) for k in langs]
    else:
        for field in ["lemma", "inflection"]:
            attended[field] = [gate_avg_attended(plots[k], field)
                               for k in langs]
        attended["total"] = [gate_avg_attended(plots[k], None) for k in langs]
    '''

    df = pd.DataFrame(attended)
    df["high"] = high
    df["low"] = low
    df["high_family"] = df["high"].apply(family_lookup.get)
    df["low_family"] = df["low"].apply(family_lookup.get)
    df["family"] = df["high_family"]
    df.loc[df["family"] != df["low_family"], "family"] = "unrelated"
    # df.loc[df["high_family"] == df["low_family"]] = df["high_family"]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-double", nargs="*")
    parser.add_argument("-gate", nargs="*")
    opt = parser.parse_args()
    assert opt.double is not None or opt.gate is not None

    
    plots = dict()
    if opt.double is not None:
        double_plots = load_data(opt.double)
        #double_frame = attn_frame(double_plots, False)
        #frames.append(double_frame)
        plots["double_plots"] = double_plots
    if opt.gate is not None:
        gate_plots = load_data(opt.gate)
        #gate_frame = attn_frame(gate_plots, True)
        #frames.append(gate_frame)
        plots["gate_plots"] = gate_plots
    #df = pd.concat(frames, axis=1)
    df = attn_frame(**plots)
        
    print(df.mean().to_latex(float_format='%.6f'))

    attn_by_family = df.groupby("low_family").mean()
    language_counts = df["low_family"].value_counts()
    attn_by_family["n_languages"] = language_counts
    print(language_counts)
    print(language_counts.sum())
    print(attn_by_family.to_latex(float_format='%.2f'))

    # ggplot(aes(x="lemma", y="inflection"), data=df) + geom_point()


if __name__ == "__main__":
    main()
