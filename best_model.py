#!/usr/bin/env python

"""
Returns the last saved model, which is also the best model if you only save
a model when it is better than all previous ones
"""

import sys
from os.path import basename, splitext, join
from glob import glob


def epoch(path):
    return int(splitext(basename(path))[0].split("_")[-1])


print(max(sys.argv[1:], key=epoch))
