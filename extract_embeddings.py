import argparse
from os.path import join

import torch

import onmt
import onmt.model_builder
import onmt.opts

from onmt.utils.misc import use_gpu


def write_embeddings(outdir, name, vocab, embeddings):
    filename = "{}_{}_embeddings.txt".format(*name)
    outpath = join(outdir, filename)
    # they're actually not that useful in this format. I want plots.
    with open(outpath, 'wb') as f:
        for i in range(min(len(embeddings), len(vocab))):
            str = vocab.itos[i].encode("utf-8")
            for j in range(len(embeddings[0])):
                str = str + (" %5f" % (embeddings[i][j])).encode("utf-8")
            f.write(str + b"\n")


def load_embeddings(path, gpu=False, **kwargs):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

    fields = checkpoint['vocab']
    model_opt = checkpoint['opt']

    for k, v in kwargs.items():
        if k not in model_opt:
            model_opt.__dict__[k] = v

    model = onmt.model_builder.build_model(model_opt, fields, gpu, checkpoint)
    encoder = model.encoder
    decoder = model.decoder
    inflection_encoder = model.inflection_encoder
    # the decoder's embedding matrix is called embeddings. fix later.
    emb = {"src": encoder.embedding, "tgt": decoder.embeddings,
           "inflection": inflection_encoder.embedding}
    result = dict()

    for field_name, multi_embedding in emb.items():
        vocab = fields[field_name][0][1].vocab
        for name, embedding_matrix in multi_embedding.items():
            embedding_matrix = embedding_matrix.weight.detach().numpy()
            result[field_name, name] = embedding_matrix, vocab

    return result


def main():
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-model', required=True, help='Path to model .pt file')
    parser.add_argument('-output_dir', default='.',
                        help="""Path to output the embeddings""")
    parser.add_argument('-gpu', type=int, default=-1, help="Device to run on")
    '''
    dummy_parser = argparse.ArgumentParser(description='train.py')
    onmt.opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]
    '''
    opt = parser.parse_args()
    gpu = use_gpu(opt)

    matrices = load_embeddings(opt.model, gpu=gpu)

    for name, (matrix, vocab) in matrices.items():
        write_embeddings(opt.output_dir, name, vocab, matrix)


if __name__ == "__main__":
    main()
