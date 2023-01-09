import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import tensorflow as tf

# -----------------------------------------------------------------------------
"""
Taken from as i am converting minGPT for TF https://github.com/karpathy/minGPT/blob/master/mingpt/utils.py
"""


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_logging(config):
    """ monotonous bookkeeping """
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))


class CfgNode:
    """ a lightweight configuration class inspired by yacs """

    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return {k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items()}

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].
        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:
        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval  # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:]  # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)


def get_padding_mask(seq):
    with tf.name_scope("Padding_Mask"):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def attention_mask(size):
    """
    if size is 4 then it returns below matrix
       [[0., 1., 1., 1.],
        [0., 0., 1., 1.],
        [0., 0., 0., 1.],
        [0., 0., 0., 0.]]
    """
    with tf.name_scope("attention_mask"):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)


def create_masks(inp):
    with tf.name_scope("att_masking"):
        att_mask = attention_mask(tf.shape(inp)[1])
        padding_mask = get_padding_mask(inp)
        mask = tf.maximum(padding_mask, att_mask)

        return mask


def top_k_logits(logits, k):
    if k == 0:
        return logits

    values, _ = tf.nn.top_k(logits, k=k)
    min_values = values[:, -1]

    return tf.where(
        logits < min_values,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits
    )


def get_weights_by_name(model, name):
    return [w for w in model.weights if w.name == name][0]


def get_perplexity(cross_entropy):
    return tf.exp(cross_entropy)


def get_default_train_config():
    C = CfgNode()
    # either model_type or (n_layer, n_head, n_embd) must be given in the config
    C.num_workers = 4
    # optimizer parameters
    C.max_iters = None
    C.batch_size = 64
    C.learning_rate = 3e-4
    C.beta_1 = 0.9
    C.beta_2 = 0.95
    C.weight_decay = 0.1  # only applied on matmul weights
    C.clip_value = 1.0
    return C
