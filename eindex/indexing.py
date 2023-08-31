import torch
from typing import Union, List
from collections import defaultdict

from ._parsing import parse_string

def eindex(
    *tensors_and_pattern: Union[str, torch.Tensor],
    **kwargs,
):
    '''
    Indexing inspired by einops notation: https://einops.rocks/

    
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
    assert isinstance(arr, torch.Tensor), "First argument must be a tensor."
    assert all(isinstance(i, torch.Tensor) for i in index_tensor_list), "All indices must be tensors."
    assert isinstance(pattern, str), "Last argument must be a string."

    # Parse the pattern string into a list of strings and lists
    #   Example #1:  ['batch', 'seq', ['batch', 'seq']]
    #   Example #2a: ['batch', 'seq', ['batch', 'seq', '0'], ['batch', 'seq', '1']]
    pattern_indices = parse_string(pattern)
    pattern_indices_str: List[str] = [p for p in pattern_indices if isinstance(p, str)]
    pattern_indices_list: List[List[str]] = [p for p in pattern_indices if isinstance(p, list)]

    # Check the dimensions are appropriate
    assert len(pattern_indices) == arr.ndim, "Invalid indices. There should be as many terms (strings or square bracket expressions) as there are dims in the first argument (arr)."

    # Check whether you're doing #2a (using a single index with multiple slices) or #2b (using multiple indices), but not both!
    using_multiple_indices = len(index_tensor_list) > 1
    using_multiple_slices = any((isinstance(i, list) and any(j.isdigit() for j in i)) for i in pattern_indices)
    assert not (using_multiple_indices and using_multiple_slices), "You can't use both multiple indices and multiple slices. Choose one or the other."

    # Create a dicionary mapping names of dimensions to their sizes (purely based on the things that appear in square brackets)
    #   Example #1: ['batch', 'seq', ['batch', 'seq']] -> {'batch': batch_size, 'seq': seq_len}
    #   Example #4: ['batch', ['batch', 'seq', 'k']] -> {'batch': batch_size, 'seq': seq_len, 'k': k}
    output_dim_counter = 0
    index_tensor_counter = 0
    dimension_sizes = {}
    for item in pattern_indices:

        # If the item is a string, we just add a single dimension: that of `arr`
        if isinstance(item, str):
            dimension_size = arr.shape[output_dim_counter]
            assert dimension_sizes.get(item, dimension_size) == dimension_size, \
                f"Incompatible dimensions. You've used {item!r} in 2 square bracket expressions, and it has 2 different values in those expressions."
            dimension_sizes[item] = dimension_size
            output_dim_counter += 1

        # If the item is a list, we add multiple dimensions: all those of the appropriate index tensor
        elif isinstance(item, list):
            # Check this square brackets expression matches the indexing tensor that it corresponds to
            assert len(item) == len(index_tensor_list[index_tensor_counter].shape), \
                "Invalid indices. There should be as many terms in each square brackets expression as the corresponding indexing tensor has dimensions."
            # Once you've asserted that it does, add the dimension sizes to the dictionary (checking for contradictions)
            for dimension_name, dimension_size in zip(item, index_tensor_list[index_tensor_counter].shape):
                assert dimension_sizes.get(dimension_name, dimension_size) == dimension_size, \
                    f"Incompatible dimensions. You've used {dimension_name!r} in 2 square bracket expressions, and it has 2 different values in those expressions."
                if not dimension_name.isdigit():
                    dimension_sizes[dimension_name] = dimension_size
            # If >1 index tensor is being used (e.g. #2b), increment the counter so we compare the next square brackets expression to the right indexing tensor
            if using_multiple_indices:
                index_tensor_counter += 1
        
    if verbose:
        print("Dimension sizes:\n  " + "\n  ".join([f"{k}: {v}" for k, v in dimension_sizes.items()]))


    # Get dimensions of output, so we know what to broadcast our indices to (when they're strings). This is all the string expressions (added the first time 
    # they appear), plus the terms in square brackets which don't also appear in string expressions (e.g. as in #4).
    #   Example #1: ['batch', 'seq', ['batch', 'seq']] -> ['batch', 'seq']
    #   Example #4: ['batch', ['batch', 'seq', 'k']] -> ['batch', 'seq', 'k']
    output_dims = []
    output_shape = []
    for item in pattern_indices:
        if isinstance(item, str):
            output_dims.append(item)
            output_shape.append(dimension_sizes[item])
        elif isinstance(item, list):
            for dim_name in item:
                if (dim_name not in pattern_indices_str) and (dim_name not in output_dims) and not(dim_name.isdigit()):
                    output_dims.append(dim_name)
                    output_shape.append(dimension_sizes[dim_name])
    output_shape = tuple(output_shape)
    output_ndim = len(output_shape)
        

    # Start constructing the index `full_idx`, by appending t.arange objects or tensors to it
    output_dim_counter = 0
    index_tensor_counter = 0
    full_idx = []
    for (item, dim_size) in zip(pattern_indices, arr.shape):

        # ! If item in pattern string is just a str, we put a rearranged indices tensor here so it can broadcast with the index tensors
        #   Example #1: 'batch' -> torch.arange(batch_size).reshape(batch_size, 1)
        #   Example #1: 'seq'   -> torch.arange(seq_len).reshape(1, seq_len)
        #   Example #4: 'batch' -> torch.arange(batch_size).reshape(batch_size, 1, 1)
        
        if isinstance(item, str):
            shape = [1] * output_ndim
            shape[output_dim_counter] = dim_size
            output_dim_counter += 1
            # ! Append the broadcasted torch.arange object to the full list of indices
            full_idx.append(torch.arange(dim_size).reshape(*shape))
        
        # ! If item is a list, this means we should be indexing into the corresponding index tensor
        # i.e. we need to construct `idx` and `shape` s.t. index_tensor[idx].reshape(shape) is the thing we want to append to the `full_idx` list
        #   Example #1:  ['batch', 'seq']      -> we want indices[:, :]
        #   Example #2a: ['batch', 'seq', '0'] -> we want indices[:, :, 0]
        #   Example #3:  ['batch']             -> we want indices[:].reshape(batch_size, 1) so it broadcasts with the final output shape
        
        elif isinstance(item, list):
        
            # Get the correct indices tensor, and increment the counter if we're using multiple indices
            index_tensor = index_tensor_list[index_tensor_counter]
            if using_multiple_indices:
                index_tensor_counter += 1
        
            # Get `idx` we'll be using to index our tensor - this will involve either slice(None) or taking a slice e.g. in Example #2a
            idx = [int(i) if i.isdigit() else slice(None) for i in item]
        
            # Get `shape` we'll be using to broadcast our tensor back out to the shape of final output. This is necessary if the indexing
            # tensor has fewer dims than the final output, e.g. Example #3. We do this by taking the output shape, and replacing the elements
            # with 1s if they're not dimensions that already exist in `index_tensor[idx]`.
            shape = list(output_shape)
            for dim_idx, dim_name in enumerate(output_dims):
                if dim_name not in item:
                    shape[dim_idx] = 1
            assert torch.tensor(shape).prod().item() == index_tensor[idx].numel(), \
                "Something's gone wrong with the shape broadcasting. Please submit an issue at https://github.com/callummcdougall/eindex"
            # ! Append the reshaped index tensor to the full list of indices
            full_idx.append(index_tensor[idx].reshape(*shape))

    return arr[full_idx]
