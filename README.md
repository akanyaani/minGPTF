# minGPTF

Originally implemented in pytorch by Andrej Karpathy :- ["karpathy/minGPT"](https://github.com/karpathy/minGPT). 


**A Tensorflow re-implementation of minGPT, **
If you guys want to explore proper implemenation of gpt2 in tensorflow have a look at ["akanyaani/gpt-2-tensorflow2.0"](https://github.com/akanyaani/gpt-2-tensorflow2.0)

**Setup**

```
$ git clone https://github.com/akanyaani/minGPTF
$ cd minGPTF
$ pip setup.py install
```

**Usage**

For generating text using GPT2
```
$ open generate.ipynb
```

Here's how you'd instantiate a GPT-2 (124M param version):

```
$ from mingptf.model import GPT
$ model_config = GPT.get_default_config()
$ model_config.vocab_size = 50257 # openai's model vocabulary
$ model_config.block_size = 1024  # openai's model block_size (i.e. input context length)
$ model = GPT(model_config)
```

And here's how you'd train it:
```
$ from mingptf.model import GPT
$ model_config = GPT.get_default_config()

$ model_config.model_type = 'gpt-micro'
$ model_config.vocab_size = 50257
$ model_config.block_size = 128
$ model = GPT(model_config)

$ train_config = get_default_train_config()
$ train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
$ train_config.max_iters = 2000

$ model.configure_optimizers(train_config)
$ model.fit(train_data, test_data, test_freq=5)
```

TO DO
```
1. Tensorfboard loging
3. Mixed precison training
4. Fine-Tuning wrapper.
```

**References:**

* ["karpathy/minGPT"](https://github.com/karpathy/minGPT)
* ["akanyaani/gpt-2-tensorflow2.0"](https://github.com/akanyaani/gpt-2-tensorflow2.0)
* ["Openai/gpt-2"](https://github.com/openai/gpt-2)
* ["Huggingface pytorch-transformers"](https://github.com/huggingface/pytorch-transformers)
* ["Tensorflow Transformers"](https://www.tensorflow.org/beta/tutorials/text/transformer)
* ["The Illustrated GPT-2 "](https://jalammar.github.io/illustrated-gpt2/)
* 
**Author**

* Abhay Kumar
* Author Email : akanyaani@gmail.com
* Follow me on [Twitter](https://twitter.com/akanyaani)

**License**

* [MIT](https://github.com/akanyaani/gpt-2-tensorflow2.0/blob/master/LICENSE)
