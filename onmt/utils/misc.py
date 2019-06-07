# -*- coding: utf-8 -*-


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, 'gpu_ranks') and len(opt.gpu_ranks) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)


def side_information(batch):
    """
    inflection, inflection_lengths = batch.inflection
    side_info = {"inflection": inflection,
                 "inflection_lengths": inflection_lengths}
    if hasattr(batch, "language"):
        side_info["language"] = batch.language
    """
    side_info = dict()
    for k in batch.fields:
        if k != "src" and k != "tgt":
            v = batch.__dict__[k]
            if isinstance(v, tuple):
                side_info[k] = v[0]
                side_info[k + "_lengths"] = v[1]
            else:
                side_info[k] = v
    return side_info
