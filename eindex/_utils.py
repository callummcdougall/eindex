import re
from typing import List

def label_dimension(pattern_and_dimensions_string: str, dimension: str, value: int) -> str:
    '''
    pattern_and_dimensions_string starts as the user input string, e.g. something like "batch seq [batch seq]".

    This function incrementally updates its labels, e.g. "batch=32 seq [batch seq]".

    When there's an "incompatible dimensions" error, it's helpful to print this string out.
    '''
    pattern_and_dimensions_string = re.sub(
        f"({dimension})" + r'(?=[\] ]|$)', # First instance of this dimension which isn't already labelled
        lambda x: x.group(1) + '=' + str(value), # Label this dimension with `value`
        pattern_and_dimensions_string,
        count = 1
    )
    return pattern_and_dimensions_string



def check_dimension_compatability(pattern_and_dimensions_string: str, dimensions_list: List[str]) -> None:
    '''
    Checks for incompatible dimensions, once all of them have been labelled.
    '''
    for dimension in dimensions_list:

        previous_dim_values = re.findall(f'{dimension}' + r'=(\d+)', pattern_and_dimensions_string)
        previous_dim_values = [int(x) for x in previous_dim_values]
        assert all([x == previous_dim_values[0] for x in previous_dim_values]), \
            f'Incompatible sizes for {dimension!r} dimension.\n' \
            + f'Based on your inputs, the inferred dimension sizes are {pattern_and_dimensions_string!r}.\n' \
            + f'Note - inputs in square brackets are inferred from the index tensor dimensions; inputs not in square brackets are inferred from the first tensor\'s dimensions.'