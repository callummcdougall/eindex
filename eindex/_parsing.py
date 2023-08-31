import re
from typing import List, Union

def parse_string(input_string: str) -> List[Union[str, List[str]]]:
    # Check for invalid characters
    if re.search(r'[^a-zA-Z0-9_\[\] ]', input_string):
        raise ValueError("Invalid characters detected in the string.")
    
    # Split the string into segments based on square brackets or spaces
    segments = re.findall(r'\[([a-zA-Z0-9_ ]+?)\]|([a-zA-Z0-9_]+)', input_string)
    
    # Process segments into the desired output format
    result = []
    for segment in segments:
        # If segment contains square brackets
        if segment[0]:
            result.append(segment[0].split())
        # If segment is outside square brackets
        elif segment[1]:
            result.append(segment[1])
    
    return result
