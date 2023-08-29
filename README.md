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

See [my blog](https://www.perfectlynormal.co.uk/blog-eindex) for more on how to use this.
