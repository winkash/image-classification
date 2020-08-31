from itertools import chain


def batch_list(input_list, batch_size=10):
    """
    Returns new batched list where each item is of size <= batch_size

    Args:
        input_list -- list of arbitrary size
    Kwargs:
        batch_size -- required batch size in output list
    """
    batched_list = []
    for i in xrange(0, len(input_list), batch_size):
        batched_list.append(input_list[i:i+batch_size])
    return batched_list

def merge_list(input_list):
    """
    Flattens the input_list by one dimension
    """
    return [i for i in chain(*input_list)]
