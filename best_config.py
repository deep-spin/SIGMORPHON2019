#!/usr/bin/env python

import sys
from glob import glob
from os.path import join, splitext, basename


def acc(line):
    return float(line.strip().split()[-1])


def max_acc(log_path):
    with open(log_path) as f:
        return max(acc(line) for line in f if "valid accuracy" in line)


def epoch(model_path):
    return int(splitext(basename(model_path))[0].split("_")[-1])


if __name__ == "__main__":
    log_files = glob(join(sys.argv[1], "*.log"))
    best_log = max(log_files, key=max_acc)
    model_dir = splitext(best_log)[0] + "-models"
    models = glob(join(model_dir, "*.pt"))
    # this works because models are only saved if they are the best so far:
    best_model = max(models, key=epoch)
    print(best_model)
