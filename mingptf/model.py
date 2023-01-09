import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization
import numpy as np

from mingptf.utils import CfgNode as CN
from mingptf.utils import create_masks, top_k_logits, get_weights_by_name
from copy import copy
from abc import ABC


def gelu(x):
    """
    Reference:- https://arxiv.org/abs/1606.08415
    """
    with tf.name_scope("gelu"):
        cdf = 0.5 * (1.0 + tf.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf


class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projection weights
        self.c_attn = tf.keras.layers.Dense(config.n_embd * 3, name="c_attn")
        self.c_proj = tf.keras.layers.Dense(config.n_embd, name="c_proj")
        # Dropout Layers
        self.attn_dropout = tf.keras.layers.Dropout(config.attn_pdrop)
        self.resid_dropout = tf.keras.layers.Dropout(config.resid_pdrop)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def call(self, x, mask):
        B, T, C = tf.shape(x)  # Batch_size, Seq_len, Embedding_size

        self.depth = C // self.n_head

        x = self.c_attn(x)
        q, k, v = tf.split(x, 3, axis=2)

        # Splitting Heads
        q = tf.transpose(tf.reshape(q, (B, T, self.n_head, self.depth)), perm=[0, 2, 1, 3])  # (B, nh, T, hs)
        k = tf.transpose(tf.reshape(k, (B, T, self.n_head, self.depth)), perm=[0, 2, 1, 3])  # (B, nh, T, hs)
        v = tf.transpose(tf.reshape(v, (B, T, self.n_head, self.depth)), perm=[0, 2, 1, 3])  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = tf.matmul(q, k, transpose_b=True) * (1.0 / tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32)))
        att += (mask * -1e9)
        # print(att)
        att = tf.nn.softmax(att, axis=-1)  # (..., seq_len_q, seq_len_k)
        # print(att)
        att = self.attn_dropout(att)
        y = tf.matmul(att, v)
        y = tf.reshape(tf.transpose(y, perm=[0, 2, 1, 3]), (B, T, C))  # Merging Heads

        # Output Projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(tf.keras.layers.Layer):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = tf.keras.layers.Dense(4 * config.n_embd, name="c_fc")
        self.c_proj = tf.keras.layers.Dense(config.n_embd, name="c_proj")
        self.dropout = tf.keras.layers.Dropout(config.resid_pdrop)

    def call(self, x):
        return self.dropout(self.c_proj(gelu(self.c_fc(x))))


class Block(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln_1 = LayerNormalization(name="ln_1")
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNormalization(name="ln_2")

        self.mlp = MLP(config)

    def call(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.wpe = tf.keras.layers.Embedding(config.block_size, config.n_embd, name="wpe")  # Position Embeddings
        self.wte = tf.keras.layers.Embedding(config.vocab_size, config.n_embd, name="wte")  # Word Embeddings

        self.drop = tf.keras.layers.Dropout(config.embd_pdrop)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNormalization(name="ln_f")


class GPT(tf.keras.Model, ABC):

    def __init__(self, config):
        super(GPT, self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.config = config

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])

        assert type_given ^ params_given  # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                                       # names follow the huggingface naming conventions
                                       # GPT-1
                                       'openai-gpt': dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                                       # GPT-2 configs
                                       'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                                       'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
                                       'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
                                       'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
                                       # Gophers
                                       'gopher-44m': dict(n_layer=8, n_head=16, n_embd=512),
                                       # (there are a number more...)
                                       # I made these tiny models up
                                       'gpt-mini': dict(n_layer=6, n_head=6, n_embd=192),
                                       'gpt-micro': dict(n_layer=4, n_head=4, n_embd=128),
                                       'gpt-nano': dict(n_layer=3, n_head=3, n_embd=48),
                                   }[config.model_type])

        self.transformer = Transformer(config)
        self(np.array([[1, 3, 5], [2, 3, 4]]))  # Passing dummy input
        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = np.sum([np.prod(v.get_shape().as_list()) for v in self.weights[:-1]])
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def call(self, x):
        mask = create_masks(x)
        """
        if seq len is 4 then mask looks like this
               [[0., 1., 1., 1.],
                [0., 0., 1., 1.],
                [0., 0., 0., 1.],
                [0., 0., 0., 0.]]
        """

        b, t = tf.shape(x)

        pos = tf.expand_dims(tf.range(0, t), 0)
        tok_emb = self.transformer.wte(x)  # Converting ids to word embeddings
        pos_emb = self.transformer.wpe(pos)  # Converting position ids to position embeddings
        x = self.transformer.drop(tok_emb + pos_emb)  # Embeddings Dropout

        for block in self.transformer.h:
            x = block(x, mask)
        x = self.transformer.ln_f(x)  # Applying layer Norm

        h_flat = tf.reshape(x, [-1, self.config.n_embd])
        logits = tf.reshape(tf.matmul(h_flat, self.transformer.wte.weights, transpose_b=True),
                            x.get_shape().as_list()[:-1] + [
                                self.config.vocab_size])  # Using Embeddings weights for vocab projection

        return logits

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import TFGPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257  # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)

        # init a huggingface/transformers model
        model_hf = TFGPT2LMHeadModel.from_pretrained(model_type)

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        hf_keys = [k.name for k in model_hf.weights if not k.name.endswith('attn/masked_bias')]  # ignore these
        md_keys = [k.name for k in model.weights]
        transposed = ['c_attn/bias', 'c_proj/bias', 'c_fc/bias', 'c_proj/bias']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Dense.

        assert len(md_keys) == len(hf_keys)
        for hf, md in zip(*(hf_keys, md_keys)):
            if any(hf.endswith(w + ":0") for w in transposed):
                assert get_weights_by_name(model_hf, hf).shape[1] == get_weights_by_name(model, md).shape
                get_weights_by_name(model, md).assign(get_weights_by_name(model_hf, hf).numpy()[0])
            else:
                # vanilla copy over the other parameters
                assert get_weights_by_name(model_hf, hf).shape == get_weights_by_name(model, md).shape
                get_weights_by_name(model, md).assign(get_weights_by_name(model_hf, hf).numpy())

        return model

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()

        blacklist_weight_modules = (LayerNormalization, tf.keras.layers.Embedding)

        for layer in self.layers:
            for var in layer.trainable_variables:
                if var.name.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(var.name)
                elif var.name.endswith('weight') and isinstance(layer, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(var.name)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.trainable_variables()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        optimizer = tf.keras.optimizers.experimental.AdamW(
            lr=train_config.learning_rate, weight_decay=train_config.weight_decay, beta_1=train_config.beta_1,
            beta_2=train_config.beta_2
        )

        optimizer.exclude_from_weight_decay(var_list=list(blacklist_weight_modules))
        return optimizer

    def generate(self, idx, max_new_tokens=512, temperature=1, top_k=8):

        for i in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :] / tf.cast(temperature, tf.float32)
            logits = top_k_logits(logits, k=top_k)
            samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
            idx = tf.concat([idx, samples], axis=-1)
        result = tf.squeeze(idx, axis=0)
        # generated_seq = tokenizer.decode(result.numpy())
        return result

# model = GPT.from_pretrained('gpt2')
