import re

def parse_string(input_string):
    '''
    Parses the input string argument, and returns a list of dimension names & offsets.

    Examples:

        "batch seq [batch seq]" -> (['batch', 'seq', ['batch', 'seq']], [0, 0, [0, 0]])

        "batch seq [batch seq+1]" -> (['batch', 'seq', ['batch', 'seq']], [0, 0, [0, 1]])
    
    Also checks for an einops operation (see docstring of main `eindex` function).
    '''
    # Check for an einops operation
    einops_operation = None
    if " -> " in input_string:
        input_string, einops_operation = input_string.split(" -> ")[:2]

    # Check for invalid characters
    if re.search(r'[^a-zA-Z0-9\+_\[\] ]', input_string):
        raise ValueError("Invalid characters detected in the string.")
    
    # Split the string into segments based on square brackets or spaces
    # This will be a list of tuples:
    # - first element is nonempty when there's a square bracket expression,
    # - second element is nonempty when there's a non-square-bracket expression
    segments = re.findall(r'\[([a-zA-Z0-9\+_ ]+?)\]|([a-zA-Z0-9\+_]+)', input_string)
    
    # Process segments into the desired output format
    # We also extract the offsets, i.e. "seq+1" would give us a result of "seq" and an offset of 1
    result = []
    offsets = []
    for segment in segments:
        # If segment contains square brackets
        if segment[0]:
            next_result = [x.split('+')[0] for x in segment[0].split()]
            next_offset = [(int(x.split('+')[1]) if '+' in x else 0) for x in segment[0].split()]
            result.append(next_result)
            offsets.append(next_offset)
        # If segment is outside square brackets
        elif segment[1]:
            result.append(segment[1].split('+')[0])
            offsets.append(int(segment[1].split('+')[1]) if '+' in segment[1] else 0)
    
    return result, offsets, einops_operation