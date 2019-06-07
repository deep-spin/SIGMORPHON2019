#!/usr/bin/env python

import argparse
from itertools import cycle
from collections import defaultdict
import pandas as pd


def config_table(df):
    result = dict()
    for config in sorted(df["config"].unique()):
        #if not config.endswith("baseline"):
        config_frame = df.loc[df["config"] == config].set_index("pair")
        config_frame = config_frame.drop("config", axis=1)
        config_frame = config_frame.drop("levenshtein", axis=1)
        result[config] = config_frame.squeeze()
    df = pd.DataFrame(result)
    best = df.max(axis=1)
    worst = df.min(axis=1)
    best_config = df.idxmax(axis=1)
    worst_config = df.idxmin(axis=1)
    df["best"] = best
    df["worst"] = worst
    df["best_config"] = best_config
    df["worst_config"] = worst_config
    df["range"] = best - worst
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="output of translate_and_evaluate.sh")
    parser.add_argument("out", help="path to save to")
    parser.add_argument("-has_config", action="store_true")
    opt = parser.parse_args()

    if opt.has_config:
        line_order = cycle(["pair", "config", "path", "eval_out"])
    else:
        line_order = cycle(["pair", "path", "eval_out"])
    line_dict = defaultdict(list)
    with open(opt.results) as f:
        for line in f:
            line = line.strip()
            if line.startswith("PRED") or line.startswith("GOLD"):
                continue
            label = next(line_order)
            if label == "eval_out":
                acc, lev = tuple(float(t) for t in line.split("\t")[1::2])
                line_dict["accuracy"].append(acc)
                line_dict["levenshtein"].append(lev)
            else:
                line_dict[label].append(line)
    df = pd.DataFrame(line_dict)
    df = df.drop("path", axis=1)

    surprise = ["danish--yiddish", "dutch--yiddish", "estonian--karelian",
                "finnish--karelian", "german--middle-high-german",
                "german--middle-low-german", "german--yiddish",
                "greek--bengali", "hungarian--karelian", "persian--azeri",
                "persian--pashto", "zulu--swahili"]
    
    df = df.loc[~df["pair"].isin(surprise)]
    df["config"] = "hi"
    configs = config_table(df)
    # print(configs.sort_values(by="range"))
    print(configs.mean())
    configs.to_csv(opt.out, sep="\t")


if __name__ == "__main__":
    main()
