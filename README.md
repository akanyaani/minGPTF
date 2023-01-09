# minGPTF

Originally implemented in pytorch by Andrej Karpathy :- ["karpathy/minGPT"](https://github.com/karpathy/minGPT). 


**A Tensorflow re-implementation of minGPT, **

**Setup**

```
$ git clone https://github.com/akanyaani/minGPTF
$ cd minGPTF
$ pip setup.py install
```

**Usage**

Here's how you'd instantiate a GPT-2 (124M param version):

```
$ from mingptf.model import GPT
$ model_config = GPT.get_default_config()
$ model_config.vocab_size = 50257 # openai's model vocabulary
$ model_config.block_size = 1024  # openai's model block_size (i.e. input context length)
$ model = GPT(model_config)
```

**License**

* [MIT](https://github.com/akanyaani/gpt-2-tensorflow2.0/blob/master/LICENSE)
