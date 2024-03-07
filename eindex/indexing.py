import numpy as np
import re
import torch
from typing import Union, List, overload
import einops

Arr = np.ndarray

from ._parsing import parse_string
from ._utils import label_dimension, check_dimension_compatability

# Type signature: eindex supports having first argument be a tensor and subsequent array arguments being either tensors
# or numpy arrays (because we can index into a tensor with a numpy array). The last argument must be a string.

@overload
def eindex(
    *tensors_and_pattern: Union[str, Arr],
    **kwargs,
) -> Arr:
    ...

@overload
def eindex(
    first_tensor: torch.Tensor,
    *tensors_and_pattern: Union[str, Arr, torch.Tensor],
    **kwargs,
) -> torch.Tensor:
    ...

def eindex(
    *tensors_and_pattern: Union[str, Union[Arr, torch.Tensor]],
    **kwargs,
) -> Union[Arr, torch.Tensor]:
    '''
    Indexing inspired by einops notation: https://einops.rocks/

    See colab for test cases: 

    
    ==========================================================================================
    ======================================== EXAMPLES ========================================
    ==========================================================================================

    I've given 5 examples here. Most of the code below is annotated with explanations relating it
    to one or more of these examples, to help explain what's going on.

    Some examples:

        # (1) You want to get all the logprobs for correct tokens

            output[batch, seq] = logrobs[batch, seq, labels[batch, seq]]

            pattern = "batch seq [batch seq]"

        # (2a) Same, but d_vocab_out is 2D not 1D for some reason
            
            output[batch, seq] = logrobs[batch, seq, labels[batch, seq, 0], labels[batch, seq, 1]]

            pattern = "batch seq [batch seq 0] [batch seq 1]"
        
        # (2b) Using 2 tensors
            
            output[batch, seq] = logrobs[batch, seq, labels_1[batch, seq], labels_2[batch, seq]]

            pattern = "batch seq [batch seq] [batch seq]"

        # (3) You want to get the logit lens at particular sequence positions (a different seq pos
              for each sequence in the batch)
        
            output[batch, d_vocab] = logprobs[batch, labels[batch], d_vocab]
            
            pattern = "batch [batch] d_vocab"

        # (4) You're indexing into a 2D tensor of tokens, for each destination token you're trying to get 5 source tokens
              (i.e. indices[batch, seqQ, :] = the seqK positions of the top 5 source tokens)
        
            output[batch, seqQ, k] = tokens[batch, indices[batch, seqQ, k]]

            pattern = "batch [batch seqQ k]"

        # (5) Same as (1), except you're producing an array of shape (batch, seq-1) cause this is before slicing.

            output[batch, seq] = logrobs[batch, seq, labels[batch, seq+1]]

            pattern = "batch seq [batch seq+1]"

        # (6) If we want to rearrange the output, we can append -> onto the end. Functionally, this just adds an einops
              rearrange operation. It's only necessary when the order of appearance of the dimensions in the pattern
              doesn't match what you want the final shape to be.

            output[seq, batch] = logrobs[batch, seq, labels[batch, seq]]

            pattern = "batch seq [batch seq] -> seq batch"

            
    ==========================================================================================
    ======================== ROUGH SUMMARY OF HOW THIS FUNCTION WORKS ========================
    ==========================================================================================

    This is implemented explicitly, i.e. with no built-in PyTorch functions like `gather` (cause they confuse me lol). 
    For example, the way we implement (1) is by creating an index like this:
    
        logprobs[
            torch.arange(batch_size).reshape(batch_size, 1),
            torch.arange(seq_len).reshape(1, seq_len),
            labels
        ]

    because when you index with tensors across different dimensions, they're all implicitly broadcast together. So we get:

        output[b, s] = logprobs[
            torch.arange(batch_size).reshape(batch_size, seq_len)[b, s],
            torch.arange(seq_len).reshape(batch_size, seq_len)[b, s],
            labels[b, s]
        ] = logprobs[b, s, labels[b, s]]

    You can use `verbose=True` to print out the dimensions of the shape you'll get at the end.
    '''

    # Unpack and type-check arguments
    arr, *index_tensor_list, pattern = tensors_and_pattern
    verbose = kwargs.pop("verbose", False)
    assert len(kwargs) == 0, f"Unexpected keyword arguments: {kwargs.keys()}"
    assert isinstance(pattern, str), "Last argument must be a string."
    
    # Convert everything to torch tensors (make sure that we remember whether it was originally a numpy array)
    orig_type = "torch" if isinstance(arr, torch.Tensor) else "numpy"
    arr = torch.from_numpy(arr) if orig_type == "numpy" else arr
    index_tensor_list = [torch.from_numpy(i) if isinstance(i, np.ndarray) else i for i in index_tensor_list]

    # Parse the pattern string into a list of dimension names (and a list of offsets, if they exist)
    #   Example #1:  ['batch', 'seq', ['batch', 'seq']] and [0, 0, [0, 0]]
    #   Example #2a: ['batch', 'seq', ['batch', 'seq', '0'], ['batch', 'seq', '1']] and [0, 0, [0, 0], [0, 0]]
    #   Example #5: ['batch', 'seq', ['batch', 'seq+1']] and [0, 0, [0, 1]]
    pattern_indices, pattern_offsets, einops_operation = parse_string(pattern)
    pattern_indices_str: List[str] = [p for p in pattern_indices if isinstance(p, str)]

    # Check the dimensions are appropriate
    assert len(pattern_indices) == arr.ndim, "Invalid indices.\n" + \
        f"Number of terms in your string pattern = {len(pattern_indices)}\n" + \
        f"Number of terms in your array to index into = {arr.ndim}\n" + \
        "These should match."

    # Check whether you're doing #2a (using a single index with multiple slices) or #2b (using multiple indices), but not both!
    using_multiple_indices = len(index_tensor_list) > 1
    using_multiple_slices = any((isinstance(i, list) and any(j.isdigit() for j in i)) for i in pattern_indices)
    assert not (using_multiple_indices and using_multiple_slices), "You can't use both multiple indices and multiple slices. Choose one or the other."

    # Create a dicionary mapping names of dimensions to their sizes (purely based on the things that appear in square brackets)
    #   Example #1: ['batch', 'seq', ['batch', 'seq']] -> {'batch': batch_size, 'seq': seq_len}
    #   Example #4: ['batch', ['batch', 'seq', 'k']] -> {'batch': batch_size, 'seq': seq_len, 'k': k}
    output_dim_counter = 0
    index_tensor_counter = 0
    pattern_and_dimensions_string = pattern # if we have incompatible dims, this will get printed out
    dimension_sizes = {}
    dimension_offset_sizes = {}
    for item, item_offset in zip(pattern_indices, pattern_offsets):

        # If the item is a string, we just add a single dimension: that of `arr`
        if isinstance(item, str):
            dimension_size = arr.shape[output_dim_counter]
            pattern_and_dimensions_string = label_dimension(pattern_and_dimensions_string, item, dimension_size)
            dimension_sizes[item] = dimension_size
            dimension_offset_sizes[item] = max(dimension_offset_sizes.get(item, 0), item_offset)

        # If the item is a list, we add multiple dimensions: all those of the appropriate index tensor
        elif isinstance(item, list):
            # Check this square brackets expression matches the indexing tensor that it corresponds to
            assert len(item) == len(index_tensor_list[index_tensor_counter].shape), \
                "Invalid indices. There should be as many terms in each square brackets expression as the corresponding indexing tensor has dimensions." + \
                f"\nSquare brackets expression: {item}" + \
                f"\nIndexing tensor shape: {index_tensor_list[index_tensor_counter].shape}"
            # Once you've asserted that it does, add the dimension sizes to the dictionary (checking for contradictions)
            for dimension_name, dimension_offset, dimension_size in zip(item, item_offset, index_tensor_list[index_tensor_counter].shape):
                pattern_and_dimensions_string = label_dimension(pattern_and_dimensions_string, dimension_name, dimension_size)
                if not dimension_name.isdigit():
                    dimension_sizes[dimension_name] = dimension_size
                    dimension_offset_sizes[dimension_name] = max(dimension_offset_sizes.get(dimension_name, 0), dimension_offset)
            # If >1 index tensor is being used (e.g. #2b), increment the counter so we compare the next square brackets expression to the right indexing tensor
            if using_multiple_indices:
                index_tensor_counter += 1
        
        output_dim_counter += 1

    # Once we've labelled everything, check if there are any errors
    check_dimension_compatability(pattern_and_dimensions_string, dimension_sizes.keys())
    if verbose:
        print("Dimension sizes:\n  " + "\n  ".join([f"{k}: {v}" for k, v in dimension_sizes.items()]))


    # Get dimensions of output, so we know what to broadcast our indices to (when they're strings). This is all the string expressions (added the first time 
    # they appear), plus the terms in square brackets which don't also appear in string expressions (e.g. as in #4).
    #   Example #1: ['batch', 'seq', ['batch', 'seq']] -> ['batch', 'seq']
    #   Example #4: ['batch', ['batch', 'seq', 'k']] -> ['batch', 'seq', 'k']
    #   Example #5: ['batch', 'seq', ['batch', 'seq+1']] -> ['batch', 'seq'], and the output_shape will be (batch, seq-1) not (batch, seq)
    output_dims = []
    output_shape = []
    for item in pattern_indices:
        if isinstance(item, str):
            dimension_name = item
            output_dims.append(dimension_name)
            output_shape.append(dimension_sizes[dimension_name] - dimension_offset_sizes[dimension_name])
        elif isinstance(item, list):
            for dimension_name in item:
                if (dimension_name not in pattern_indices_str) and (dimension_name not in output_dims) and not(dimension_name.isdigit()):
                    output_dims.append(dimension_name)
                    output_shape.append(dimension_sizes[dimension_name] - dimension_offset_sizes[dimension_name])
    output_shape = tuple(output_shape)
    output_ndim = len(output_shape)

    # Start constructing the index `full_idx`, by appending t.arange objects or tensors to it
    # Note, to avoid confusion:
    # - `full_idx` is the object we'll eventually use to index into our array, i.e. the first argument of `eindex`
    # - `idx` are the objects we construct to index into our indexing arrays. Usually `idx` is just "take the entire indexing array", but this isn't
    #   always the case, e.g. when the pattern string looks like [batch seq+1] (see example #5) or [batch seq 0] (see example #2a).
    index_tensor_counter = 0
    full_idx = []
    for (item, item_offset, dimemsion_size) in zip(pattern_indices, pattern_offsets, arr.shape):


        # ! If item in pattern string is just a str, we put a rearranged indices tensor here so it can broadcast with the index tensors
        #   Example #1: 'batch' -> torch.arange(batch_size).reshape(batch_size, 1)
        #   Example #1: 'seq'   -> torch.arange(seq_len).reshape(1, seq_len)
        #   Example #4: 'batch' -> torch.arange(batch_size).reshape(batch_size, 1, 1)
        #   Example #5: 'seq'   -> torch.arange(seq_len).reshape(1, seq_len-1), because this is the output shape (dim size minus dim offset size)
        
        if isinstance(item, str):
            dimension_name = item

            # True dimension size accounts for offsets, e.g. with #5 [batch seq [batch seq+1]] the output size will be (batch, seq-1)
            true_dimemsion_size = dimemsion_size - dimension_offset_sizes[dimension_name]

            # We want our shape to be [1, 1, ..., dim_size, ..., 1, 1], with the values being the range (0, 1, ..., dim_size-1)
            shape = [1] * output_ndim
            shape[output_dims.index(dimension_name)] = true_dimemsion_size
            
            # Check the shape isn't unexpected (we should have only one element of the above tensor not equal to 1)
            assert torch.tensor(shape).prod().item() == true_dimemsion_size, \
                "Something's gone wrong with the shape broadcasting. Please submit an issue at https://github.com/callummcdougall/eindex"

            # ! Append the broadcasted torch.arange object to the full list of indices
            full_idx_item = torch.arange(true_dimemsion_size).reshape(*shape)
            full_idx.append(full_idx_item)
        

        # ! If item is a list, this means we should be indexing into the corresponding index tensor
        # i.e. we need to construct `idx` and `shape` s.t. index_tensor[idx].reshape(shape) is the thing we want to append to the `full_idx` list
        #   Example #1:  ['batch', 'seq']      -> we want indices[:, :]
        #   Example #2a: ['batch', 'seq', '0'] -> we want indices[:, :, 0]
        #   Example #3:  ['batch']             -> we want indices[:].reshape(batch_size, 1) so it broadcasts with the final output shape
        #   Example #5:  ['batch' 'seq']       -> we want indices[:, 1:seq_len] because the seq offset is +1
        
        elif isinstance(item, list):
        
            # Get the correct indices tensor, and increment the counter if we're using multiple indices
            index_tensor = index_tensor_list[index_tensor_counter]
            if using_multiple_indices:
                index_tensor_counter += 1
        
            # Get `idx` we'll be using to index into our indexing tensor
            # For each object in `item`, there are 3 cases:
            #   (A) digit case, i.e. we're dealing with something like [batch seq 0]. Here, we want to index into the index tensor with a single integer.
            #   (B) vanilla case, e.g. something like [batch seq]. Here, we want to slice the entire tensor.
            #   (C) offset case, e.g. something like [batch seq+1]. Here, we want to slice the entire tensor but with an offset. The offset is determined
            #       by the size of the offset on this item (i.e. seq+1) and the maximum offset. For example, if we have [batch seq] [batch seq+1] then we
            #       would want our slices to be [:, :seq_len-1] and [:, 1:seq_len] respectively.
            # Note that (B) is a special case of (C), so we don't deal with (B) separately.
            idx = []
            for dimension_name, offset in zip(item, item_offset):
                # Case (B)
                if dimension_name.isdigit():
                    idx.append(int(dimension_name))
                # Case (C)
                else:
                    lower = offset
                    upper = (offset - dimension_offset_sizes[dimension_name]) if (offset != dimension_offset_sizes[dimension_name]) else None
                    idx.append(slice(lower, upper))
        
            # Get `shape` we'll be using to broadcast our tensor back out to the shape of final output. This is necessary if the indexing
            # tensor has fewer dims than the final output, e.g. Example #3. We do this by taking the output shape, and replacing the elements
            # with 1s if they're not dimensions that already exist in `index_tensor[idx]`.
            shape = list(output_shape)
            for dim_idx, dim_name in enumerate(output_dims):
                if dim_name not in item:
                    shape[dim_idx] = 1
            
            # Check the shape isn't unexpected (by comparing it to the shape of the index tensor)
            assert torch.tensor(shape).prod().item() == index_tensor[idx].numel(), \
                "Something's gone wrong with the shape broadcasting. Please submit an issue at https://github.com/callummcdougall/eindex"
            
            # ! Append the reshaped index tensor to the full list of indices
            full_idx_item = index_tensor[idx].reshape(*shape)
            full_idx.append(full_idx_item )

    # Index using the full array
    arr_indexed = arr[full_idx]

    # If there was an einops operation, apply it
    if einops_operation is not None:
        einops_dims = einops_operation.split(" ")
        if einops_dims == output_dims:
            # In this case, the output dimensions are already correct, and we don't need an einops operation
            pass
        else:
            assert set(einops_dims) == set(output_dims), \
                "The dimensions in your einops operation, i.e. the part after '->', don't match the inferred output dimensions of your indexing operation." + \
                f"\nInferred output dimensions: {output_dims}" + \
                f"\nYour einops operation: {einops_dims}"
            einops_operation = f"{' '.join(output_dims)} -> {einops_operation}"
            arr_indexed = einops.rearrange(arr_indexed, einops_operation)

    if orig_type == "numpy":
        arr_indexed = arr_indexed.numpy()

    return arr_indexed


# BATCH_SIZE = 32
# SEQ_LEN = 5
# D_VOCAB = 100

# logprobs = t.randn(BATCH_SIZE, SEQ_LEN, D_VOCAB).log_softmax(-1)
# labels = t.randint(0, D_VOCAB, (BATCH_SIZE, SEQ_LEN))

# output_1A = eindex(logprobs, labels, "batch seq [batch seq]")
# output_1B = eindex(logprobs, labels, "batch seq [batch seq] -> batch seq")
# output_2 = eindex(logprobs, labels, "batch seq [batch seq] -> seq batch")

# assert t.allclose(output_1A, output_2.T)
# assert t.allclose(output_1B, output_2.T)
