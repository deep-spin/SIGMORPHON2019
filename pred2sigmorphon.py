#!/usr/bin/env python

import argparse
import sys
import re

parser = argparse.ArgumentParser()
parser.add_argument('src', type=open)
parser.add_argument('pred', type=open)
opt = parser.parse_args()

for src_line, pred_line in zip(opt.src, opt.pred):
    line_fields = src_line.strip().split('\t')
    lemma, tags = line_fields[0], line_fields[-1]
    inflection = re.sub(r'\s\s+', '<space>', pred_line.strip())
    inflection = re.sub(r'\s', '', inflection)
    inflection = re.sub(r'<space>', ' ', inflection)
    sys.stdout.write('\t'.join([lemma, inflection, tags]) + '\n')
