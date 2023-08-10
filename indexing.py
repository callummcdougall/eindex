import torch
import numpy as np
from jaxtyping import Float, Int, Shaped
from ._parsing import parse_string

def eindex(
    arr: Shaped[torch.Tensor, "..."],
    indices: Int[torch.Tensor, "..."],
    pattern: str,
):
    '''
    Some examples:

        # (1) You want to get all the logprobs for correct tokens

            output[batch, seq] = logrobs[batch, seq, labels[batch, seq]]

            pattern = "batch seq [batch seq]"

        # (2) Same, but d_vocab_out is 2D not 1D for some reason
            
            output[batch, seq] = logrobs[batch, seq, labels[batch, seq, 0], labels[batch, seq, 1]]

            pattern = "batch seq [batch seq 0] [batch seq 1]"

        # (3) You want to get the logit lens at particular sequence positions (a different seq pos for each sequence in the batch)
        
            output[batch, d_vocab] = logprobs[batch, labels[batch], d_vocab]
            
            pattern = "batch [batch] d_vocab"


    This is implemented explicitly, i.e. with no built-in PyTorch functions like `gather` (cause they confuse me lol). 
    For example, the way we implement (1) is by creating an index like this:
    
        logprobs[
            t.arange(batch_size).reshape(batch_size, 1),
            t.arange(seq_len).reshape(1, seq_len),
            labels
        ]

    because when you index with tensors across different dimensions, they're all implicitly broadcast together. So we get:

        output[b, s] = logprobs[
            t.arange(batch_size).reshape(batch_size, seq_len)[b, s],
            t.arange(seq_len).reshape(batch_size, seq_len)[b, s],
            labels[b, s]
        ] = logprobs[b, s, labels[b, s]]
    '''

    pattern_indices = parse_string(pattern)

    # Check the dimensions are appropriate
    assert len(pattern_indices) == arr.ndim, "Invalid indices. There should be as many terms (strings or square bracket expressions) as there are dims in the first argument (arr)."

    full_idx = []

    # Get dimensions of output, so we know what to broadcast our indices to (when they're strings)
    pattern_indices_str = [p for p in pattern_indices if isinstance(p, str)]
    output_shape = tuple([d for d, p in zip(arr.shape, pattern_indices) if isinstance(p, str)])
    output_ndim = len(output_shape)

    # Start constructing the index
    arr_dim_counter = -1
    for (item, dim_size) in zip(pattern_indices, arr.shape):

        # If item is just the name of a dimension, we put a rearranged indices tensor here
        #   Example #1: "batch" -> we want t.arange(batch_size).reshape(batch_size, 1)
        #   Example #1: "seq"   -> we want t.arange(seq_len).reshape(1, seq_len)
        if isinstance(item, str):
            arr_dim_counter += 1
            shape = [1] * output_ndim
            shape[arr_dim_counter] = dim_size
            full_idx_item = torch.arange(dim_size).reshape(*shape)
        
        # If item is a list, this means we're indexing into the indices using this
        #   Example #1: ["batch", "seq"]      -> we want indices[:, :]
        #   Example #2: ["batch", "seq", "0"] -> we want indices[:, :, 0]
        #   Example #3: ["batch"]             -> we want indices[:].reshape(batch_size, 1) (see long comment below)
        elif isinstance(item, list):
            # Get all the "within square brackets" indices, either `:` = slice(None) or ints
            idx = [int(i) if i.isdigit() else slice(None) for i in item]
            _indices = indices[idx]
            # If the output shape is larger than the shape of `indices`, we need to reshape `indices` to add dummy dimensions.
            # e.g. in Example #3, we want our indices to eventually broadcast to our output shape of (batch, d_vocab), and since
            # `indices` only has shape (batch,), we need to add a dummy dimension for `d_vocab`. Also, we want all dimensions with
            # integers (like in Example #2) to be dummy dimensions. So we'll start with the full output shape, and then replace
            # the elements with dummy dimensions so the reshape operation doesn't change the number of elements. 
            shape = list(output_shape)
            for i, p in enumerate(pattern_indices_str):
                if p not in item:
                    shape[i] = 1
            full_idx_item = _indices.reshape(*shape)

        full_idx.append(full_idx_item)

    return arr[full_idx]
