# Reference

[**Categorical Reparameterization with Gumbel-Softmax**](https://arxiv.org/abs/1611.01144)

[**The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables**](https://arxiv.org/abs/1611.00712)

[**Semi-Supervised Learning with Deep Generative Models**](https://arxiv.org/abs/1406.5298)

[pyro example code](https://github.com/pyro-ppl/pyro/tree/dev/examples/vae)

[discrete-VAE (base code)](https://github.com/leequant761/discrete-VAE)

# Implementation

```
python main.py --kld=eric --alpha=900
```

86% accuracy with 100 labelled data ==> eric's trick is good!

```
python main.py --kld=madisson --alpha=900
```

62% accuracy with 100 labelled data

# Results

`alpha` is very important hyperparameter to tune.

It makes latent variable y to work as label in decoding.
