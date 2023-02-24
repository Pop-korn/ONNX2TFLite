""" This file contains functions shared by the various files in the 
    'conversion/builtin'directory. """

import src.err as err



def isOfSize(obj, size: int):
    """ Determine if given object with '__len__()' method if not None
        and is of given 'size'. """

    if obj is None:
        return False
    
    return len(obj) == size


def assign2DStrides(obj, strides: list[int]):
    """ Assign the 'obj' attributes 'strideH' and 'strideW' from 'strides'.
        'obj' MUST have these attributes. """
    
    if isOfSize(strides, 2):
        obj.strideH = strides[0]
        obj.strideW = strides[1]
    else:
        err.note(f"Expected 2D strides, got '{strides}'.",
                 "Leaving default values.")


