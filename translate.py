#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configargparse

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.opts as opts


def main(opt):
    translator = build_translator(opt)
    translator.translate(opt.corpora)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description='translate.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)
