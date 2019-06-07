""" Implementation of all available options """
from __future__ import print_function

import configargparse


def config_opts(parser):
    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('-save_config', '--save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    # Embedding Options
    group = parser.add_argument_group('Model-Embeddings')
    group.add('--src_word_vec_size', '-src_word_vec_size',
              type=int, default=500,
              help='Word embedding size for src.')
    group.add('--tgt_word_vec_size', '-tgt_word_vec_size',
              type=int, default=500, help='Word embedding size for tgt.')
    group.add('--word_vec_size', '-word_vec_size', type=int, default=-1,
              help='Word embedding size for src and tgt.')
    group.add("--lang_rep", "-lang_rep", choices=["token", "feature"],
              default=None, help="""Style of representation for language ID:
              'token' will result in a token at the beginning of the
              source sequence (similar to the Google zero shot language token)
              while 'feature' will concatenate a learned language vector of
              arbitrary dimension to the main embeddings at each time step.
              Using 'token' requires lang_vec_size to be the same as the
              word vec size. Default is to add no language representation.""")
    group.add("--lang_location", "-lang_location", nargs="*",
              choices=["src", "tgt", "inflection"],
              help="""Fields that should have a language representation
              added. Including 'tgt' is only compatible with using 'feature'
              for the language representation. This option is ignored if there
              is no lang_rep.""")
    group.add('--lang_vec_size', '-lang_vec_size', type=int, default=None)
    group.add('--infl_vec_size', '-infl_vec_size', type=int, default=None)

    group.add('--share_decoder_embeddings', '-share_decoder_embeddings',
              action='store_true',
              help="""Use a shared weight matrix for the input and
                       output word  embeddings in the decoder.""")
    group.add('--share_embeddings', '-share_embeddings', action='store_true',
              help="""Share the word embeddings between encoder
                       and decoder. Requires shared dictionary.""")

    # Encoder-Decoder Options
    group = parser.add_argument_group('Model- Encoder-Decoder')
    group.add('--layers', '-layers', type=int, default=-1,
              help='Number of layers in enc/dec.')
    group.add('--enc_layers', '-enc_layers', type=int, default=2,
              help='Number of layers in the encoder')
    group.add('--dec_layers', '-dec_layers', type=int, default=2,
              help='Number of layers in the decoder')
    group.add('--rnn_size', '-rnn_size', type=int, default=-1,
              help="""Size of rnn hidden states. Overwrites
                       enc_rnn_size and dec_rnn_size""")
    group.add('--enc_rnn_size', '-enc_rnn_size', type=int, default=500,
              help="""Size of encoder rnn hidden states.
                       Must be equal to dec_rnn_size except for
                       speech-to-text.""")
    group.add('--dec_rnn_size', '-dec_rnn_size', type=int, default=500,
              help="""Size of decoder rnn hidden states.
                       Must be equal to enc_rnn_size except for
                       speech-to-text.""")
    group.add('--input_feed', '-input_feed', type=int, default=1,
              help="""Feed the context vector at each time step as
                       additional input (via concatenation with the word
                       embeddings) to the decoder.""")
    group.add('--rnn_type', '-rnn_type', type=str, default='LSTM',
              choices=['LSTM', 'GRU'])
    group.add('--brnn', '-brnn', action='store_true')
    group.add('--out_bias', '-out_bias', type=str,
              choices=['none', 'single', 'multi'], default='single')

    # Attention options
    group = parser.add_argument_group('Model- Attention')
    group.add('--global_attention', '-global_attention',
              type=str, default='general', choices=['dot', 'general'],
              help="attention: dotprod or general (Luong)")
    group.add('--global_attention_function', '-global_attention_function',
              type=str, default="softmax", choices=["softmax", "sparsemax"])
    group.add("--inflection_attention", "-inflection_attention",
              action="store_true", help="""Second attention mechanism over
              inflection tags.""")
    group.add("--inflection_rnn", "-inflection_rnn", action="store_true",
              help="""Encode inflection sequence with a bilstm instead of
              just embeddings.""")
    group.add("--inflection_gate", "-inflection_gate",
              choices=["softmax", "sparsemax"], default=None,
              help="Use a learned gate to merge attentional hidden states")
    group.add("--separate_outputs", "-separate_outputs", action="store_true",
              help="""If using an attention gate, whether to use separate
              output layers for the lemma and inflection attention heads.""")
    group.add("--inflection_rnn_layers", "-inflection_rnn_layers", default=2,
              type=int, help="Layers for inflection RNN")
    group.add('--loss', '-loss', default="xent", choices=["xent", "sparsemax"],
              help="""Which function to use for generating
              probabilities over the target vocabulary (choices:
              softmax, sparsemax)""")


def preprocess_opts(parser):
    """ Pre-procesing options """
    # Data options
    group = parser.add_argument_group('Data')
    group.add('--train', '-train', nargs='+',
              help="Path to sigmorphon training data")
    group.add('--valid', '-valid', nargs='+',
              help="Path to sigmorphon validation data")
    group.add("--allow_unseen_languages", "-allow_unseen_languages",
              action="store_true", help="""Option to not filter languages
              unseen at training time from validation.""")
    group.add("--high_oversampling", "-high_oversampling", type=int, default=1,
              help="Number of times to repeat training data from HR languages")
    group.add("--low_oversampling", "-low_oversampling", type=int, default=1,
              help="Number of times to repeat training data from LR languages")

    group.add('--save_data', '-save_data', required=True,
              help="Output file for the prepared data")

    # Dictionary options, for text corpus

    group = parser.add_argument_group('Vocab')
    group.add('--src_vocab', '-src_vocab', default="",
              help="Path to source vocabulary. Format: one word per line.")
    group.add('--tgt_vocab', '-tgt_vocab', default="",
              help="Path to target vocabulary. Format: one word per line.")
    group.add('--src_vocab_size', '-src_vocab_size', type=int, default=50000,
              help="Size of the source vocabulary")
    group.add('--tgt_vocab_size', '-tgt_vocab_size', type=int, default=50000,
              help="Size of the target vocabulary")

    group.add('--src_words_min_frequency',
              '-src_words_min_frequency', type=int, default=0)
    group.add('--tgt_words_min_frequency',
              '-tgt_words_min_frequency', type=int, default=0)

    group.add('--share_vocab', '-share_vocab', action='store_true',
              help="Share source and target vocabulary")

    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    group.add('--src_seq_length', '-src_seq_length', type=int, default=50,
              help="Maximum source sequence length")
    group.add('--src_seq_length_trunc', '-src_seq_length_trunc',
              type=int, default=None,
              help="Truncate source sequence length.")
    group.add('--tgt_seq_length', '-tgt_seq_length', type=int, default=50,
              help="Maximum target sequence length to keep.")
    group.add('--tgt_seq_length_trunc', '-tgt_seq_length_trunc',
              type=int, default=None,
              help="Truncate target sequence length.")
    group.add('--multilingual', '-multilingual', action='store_true',
              help="Add a language tag (parsed from filename) to each sample")
    group.add("--inflection_field", "-inflection_field", action="store_true",
              help="Use a separate field for inflection or cat to src.")
    group.add("--lang_src", "-lang_src", action="store_true",
              help="Put language token in source sequence.")

    # Data processing options
    group = parser.add_argument_group('Random')
    group.add('--shuffle', '-shuffle', type=int, default=0)
    group.add('--seed', '-seed', type=int, default=3435, help="Random seed")

    group = parser.add_argument_group('Logging')
    group.add('--report_every', '-report_every', type=int, default=100000,
              help="Report status every this many sentences")
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    group.add('--log_file_level', '-log_file_level', type=str,
              action=StoreLoggingLevelAction,
              choices=StoreLoggingLevelAction.CHOICES,
              default="0")


def train_opts(parser):
    """ Training and saving options """

    group = parser.add_argument_group('General')
    group.add('--data', '-data', required=True,
              help="""Path prefix to the ".train.pt" and
                       ".valid.pt" file path from preprocess.py""")

    group.add('--save_model', '-save_model', default='model',
              help="""Model filename (the model will be saved as
                       <save_model>_N.pt where N is the number of steps""")

    group.add('--keep_checkpoint', '-keep_checkpoint', type=int, default=-1,
              help="""Keep X checkpoints (negative: keep all)""")

    # GPU
    group.add('--gpuid', '-gpuid', default=[], nargs='*', type=int,
              help="Deprecated see world_size and gpu_ranks.")
    group.add('--gpu_ranks', '-gpu_ranks', default=[], nargs='*', type=int,
              help="list of ranks of each process.")
    group.add('--world_size', '-world_size', default=1, type=int,
              help="total number of distributed processes.")
    group.add('--gpu_backend', '-gpu_backend',
              default="nccl", type=str,
              help="Type of torch distributed backend")
    group.add('--gpu_verbose_level', '-gpu_verbose_level', default=0, type=int,
              help="Gives more info on each process per GPU.")
    group.add('--master_ip', '-master_ip', default="localhost", type=str,
              help="IP of master for torch.distributed training.")
    group.add('--master_port', '-master_port', default=10000, type=int,
              help="Port of master for torch.distributed training.")

    group.add('--seed', '-seed', type=int, default=-1,
              help="""Random seed used for the experiments
                       reproducibility.""")

    # Init options
    group = parser.add_argument_group('Initialization')
    group.add('--param_init', '-param_init', type=float, default=0.1,
              help="""Parameters are initialized over uniform distribution
                       with support (-param_init, param_init).
                       Use 0 to not use initialization""")

    group.add('--train_from', '-train_from', default='', type=str,
              help="""If training from a checkpoint then this is the
                       path to the pretrained model's state_dict.""")
    group.add('--reset_optim', '-reset_optim', default='none',
              choices=['none', 'all', 'states', 'keep_states'],
              help="""Optimization resetter when train_from.""")

    # Pretrained word vectors
    group.add('--pre_word_vecs_enc', '-pre_word_vecs_enc',
              help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the encoder side.
                       See README for specific formatting instructions.""")
    group.add('--pre_word_vecs_dec', '-pre_word_vecs_dec',
              help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the decoder side.
                       See README for specific formatting instructions.""")
    # Fixed word vectors
    group.add('--fix_word_vecs_enc', '-fix_word_vecs_enc',
              action='store_true',
              help="Fix word embeddings on the encoder side.")
    group.add('--fix_word_vecs_dec', '-fix_word_vecs_dec',
              action='store_true',
              help="Fix word embeddings on the decoder side.")

    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add('--batch_size', '-batch_size', type=int, default=64,
              help='Maximum batch size for training')
    group.add('--valid_batch_size', '-valid_batch_size', type=int, default=32,
              help='Maximum batch size for validation')
    group.add('--epochs', '-epochs', type=int, default=13)
    group.add('--optim', '-optim', default='sgd',
              choices=['sgd', 'adagrad', 'adadelta', 'adam',
                       'sparseadam', 'adafactor'],
              help="""Optimization method.""")
    group.add('--adagrad_accumulator_init', '-adagrad_accumulator_init',
              type=float, default=0,
              help="""Initializes the accumulator values in adagrad.
                       Mirrors the initial_accumulator_value option
                       in the tensorflow adagrad (use 0.1 for their default).
                       """)
    group.add('--max_grad_norm', '-max_grad_norm', type=float, default=5,
              help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to
                       max_grad_norm""")
    group.add('--dropout', '-dropout', type=float, default=0.3,
              help="Dropout probability; applied in LSTM stacks.")
    group.add('--truncated_decoder', '-truncated_decoder', type=int, default=0,
              help="""Truncated bptt.""")
    group.add('--adam_beta1', '-adam_beta1', type=float, default=0.9,
              help="""The beta1 parameter used by Adam.
                       Almost without exception a value of 0.9 is used in
                       the literature, seemingly giving good results,
                       so we would discourage changing this value from
                       the default without due consideration.""")
    group.add('--adam_beta2', '-adam_beta2', type=float, default=0.999,
              help="""The beta2 parameter used by Adam.
                       Typically a value of 0.999 is recommended, as this is
                       the value suggested by the original paper describing
                       Adam, and is also the value adopted in other frameworks
                       such as Tensorflow and Kerras, i.e. see:
                       https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
                       https://keras.io/optimizers/ .
                       Whereas recently the paper "Attention is All You Need"
                       suggested a value of 0.98 for beta2, this parameter may
                       not work well for normal models / default
                       baselines.""")
    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add('--learning_rate', '-learning_rate', type=float, default=1.0,
              help="""Starting learning rate.
                       Recommended settings: sgd = 1, adagrad = 0.1,
                       adadelta = 1, adam = 0.001""")
    group.add('--learning_rate_decay', '-learning_rate_decay',
              type=float, default=0.5,
              help="""If update_learning_rate, decay learning rate by
                       this much if steps have gone past
                       start_decay_steps""")
    group.add("--patience", "-patience", type=int, default=0,
              help="""Epochs without improvement
              before decaying learning rate.""")
    group.add("--valid_metric", "-valid_metric", choices=["acc", "loss"],
              default="acc")

    group = parser.add_argument_group('Logging')
    group.add('--report_every', '-report_every', type=int, default=50,
              help="Print stats at this interval.")
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    group.add('--log_file_level', '-log_file_level', type=str,
              action=StoreLoggingLevelAction,
              choices=StoreLoggingLevelAction.CHOICES,
              default="0")
    group.add('--exp_host', '-exp_host', type=str, default="",
              help="Send logs to this crayon server.")
    group.add('--exp', '-exp', type=str, default="",
                       help="Name of the experiment for logging.")
    # Use TensorboardX for visualization during training
    group.add('--tensorboard', '-tensorboard', action="store_true",
              help="""Use tensorboardX for visualization during training.
                       Must have the library tensorboardX.""")
    group.add_argument("--tensorboard_log_dir", "-tensorboard_log_dir",
                       type=str, default="runs/onmt",
                       help="""Log directory for Tensorboard.
                       This is also the name of the run.""")

    group.add_argument("--no_shuffle", "-no_shuffle",
                       action="store_true", help="""Whether to keep the order
                       of examples from the (possibly multilingual) dataset.
                       This is probably a bad option to select.""")


def translate_opts(parser):
    """ Translation / inference options """
    group = parser.add_argument_group('Model')
    group.add('--model', '-model', dest='models', metavar='MODEL',
              nargs='+', type=str, default=[], required=True,
              help='Path to model .pt file(s). '
              'Multiple models can be specified, '
              'for ensemble decoding.')
    group.add('--avg_raw_probs', '-avg_raw_probs', action='store_true',
              help="""If this is set, during ensembling scores from
              different models will be combined by averaging their
              raw probabilities and then taking the log. Otherwise,
              the log probabilities will be averaged directly.
              Necessary for models whose output layers can assign
              zero probability.""")

    group = parser.add_argument_group('Data')
    group.add('--data_type', '-data_type', default="text",
              help="Type of the source input. Options: [text|img].")

    group.add('--corpora', '-corpora', required=True,
              help="Paths to SIGMORPHON files")
    group.add('--output', '-output', default='pred.txt',
              help="Path to output the predictions")
    group = parser.add_argument_group('Beam')
    group.add('--beam_size', '-beam_size', type=int, default=5)
    group.add('--max_length', '-max_length', type=int, default=100,
              help='Maximum prediction length.')
    group.add('--max_sent_length', '-max_sent_length', action=DeprecateAction,
              help="Deprecated, use `-max_length` instead")

    group = parser.add_argument_group('Logging')
    group.add('--verbose', '-verbose', action="store_true",
              help='Print scores and predictions for each sentence')
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    group.add('--log_file_level', '-log_file_level', type=str,
              action=StoreLoggingLevelAction,
              choices=StoreLoggingLevelAction.CHOICES,
              default="0")
    group.add("--attn_path", "-attn_path", default=None,
              help="""Path to output attention tensors (if None, do not save
              attention).""")
    group.add("--probs_path", "-probs_path", default=None,
              help="""Path to output vocab prob tensors (if None, do not save
              probabilities).""")
    group.add("--dump_beam", "-dump_beam", default="",
              help="""Path to output json beam representations.""")
    group.add('--n_best', '-n_best', type=int, default=1,
              help="If verbose, outputs the n_best decoded sentences")

    group = parser.add_argument_group('Efficiency')
    group.add('--batch_size', '-batch_size', type=int, default=1)
    group.add('--gpu', '-gpu', type=int, default=-1, help="Device to run on")


def add_md_help_argument(parser):
    """ md help parser """
    parser.add('--md', '-md', action=MarkdownHelpAction,
               help='print Markdown-formatted help text and exit.')


# MARKDOWN boilerplate

# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
class MarkdownHelpFormatter(configargparse.HelpFormatter):
    """A really bare-bones configargparse help formatter that generates valid
       markdown.

       This will generate something like:
       usage
       # **section heading**:
       ## **--argument-one**
       ```
       argument-one help text
       ```
    """

    def _format_usage(self, usage, actions, groups, prefix):
        return ""

    def format_help(self):
        print(self._prog)
        self._root_section.heading = '# Options: %s' % self._prog
        return super(MarkdownHelpFormatter, self).format_help()

    def start_section(self, heading):
        super(MarkdownHelpFormatter, self) \
            .start_section('### **%s**' % heading)

    def _format_action(self, action):
        if action.dest == "help" or action.dest == "md":
            return ""
        lines = []
        lines.append('* **-%s %s** ' % (action.dest,
                                        "[%s]" % action.default
                                        if action.default else "[]"))
        if action.help:
            help_text = self._expand_help(action)
            lines.extend(self._split_lines(help_text, 80))
        lines.extend(['', ''])
        return '\n'.join(lines)


class MarkdownHelpAction(configargparse.Action):
    """ MD help action """

    def __init__(self, option_strings,
                 dest=configargparse.SUPPRESS, default=configargparse.SUPPRESS,
                 **kwargs):
        super(MarkdownHelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.formatter_class = MarkdownHelpFormatter
        parser.print_help()
        parser.exit()


class StoreLoggingLevelAction(configargparse.Action):
    """ Convert string to logging level """
    import logging
    LEVELS = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET
    }

    CHOICES = list(LEVELS.keys()) + [str(_) for _ in LEVELS.values()]

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(StoreLoggingLevelAction, self).__init__(
            option_strings, dest, help=help, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        # Get the key 'value' in the dict, or just use 'value'
        level = StoreLoggingLevelAction.LEVELS.get(value, value)
        setattr(namespace, self.dest, level)


class DeprecateAction(configargparse.Action):
    """ Deprecate action """

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0,
                                              help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.mdhelp is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise configargparse.ArgumentTypeError(msg)
