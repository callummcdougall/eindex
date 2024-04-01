# eindex

My interpretation of what einops indexing would look like (created to work on during my SERI MATS project).

Install with:

```
pip install git+https://github.com/callummcdougall/eindex.git
```

Import with:

```python
from eindex import eindex
```

Use as:

```python
output = eindex(logprobs, labels, "batch seq [batch seq]")
```

which is functionally the same as setting the elements of the 2D `output` tensor as follows:

```
output[batch, seq] = logprobs[batch, seq, labels[batch, seq]]
```

See the accompanying [Colab notebook](https://colab.research.google.com/drive/1KbuRsoKTMrgjtOQgUDeam8GWX0k1YzmO?usp=sharing) (or [my blog](https://www.perfectlynormal.co.uk/blog-eindex)) for more on how to use this.

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/indexing.png" width="320">
