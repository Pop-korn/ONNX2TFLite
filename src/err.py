"""
    err

Module implements functions for logging, error messages and custom assertions.

__author__ = Martin Pavella
__version__ = 1.0
__email__ = xpavel39@stud.fit.vutbr.cz
"""

import sys
from enum import Enum

""" Minumum message importance level to print. """
class MessageImportance(Enum):
    LOWEST = -1
    UNCHECKED = 0
    INTERNAL = 1
    NOTE = 2
    WARNING = 3
    ERROR = 4
    HIGHEST = 10

MIN_OUTPUT_IMPORTANCE = MessageImportance.WARNING


class Code(Enum):
    INTERNAL_ERR = 1
    GENERATED_MODEL_ERR = 2

    INPUT_FILE_ERR = 11

    UNSUPPORTED_OPERATOR = 21
    UNSUPPORTED_ONNX_TYPE = 22
    UNSUPPORTED_OPERATOR_ATTRIBUTES = 23
    NOT_IMPLEMENTED = 24

    INVALID_TYPE = 31
    INVALID_TENSOR_SHAPE = 32
    INVALID_ONNX_OPERATOR = 33

    CONVERSION_IMPOSSIBLE = 41

    INPUT_ERR = 51

def error(errCode: Code, *args, **kwargs):
    """ Print error message with given parameters. Alse EXIT code execution but ONLY IF
        'errCode' is not None. """
    if MIN_OUTPUT_IMPORTANCE.value > MessageImportance.ERROR.value:
        return
    
    print("\tERROR: ", *args, file=sys.stderr, **kwargs)
    if errCode is not None:
        exit(errCode.value)

def warning(*args, **kwargs):
    """ Print warning message with given parameters. """

    if MIN_OUTPUT_IMPORTANCE.value > MessageImportance.WARNING.value:
        return
    
    print("\tWARNING: ", *args, file=sys.stderr, **kwargs)

def note(*args, **kwargs):
    """ Print note message with given parameters. """
    
    if MIN_OUTPUT_IMPORTANCE.value > MessageImportance.NOTE.value:
        return

    print("\tNOTE: ", *args, file=sys.stderr, **kwargs)

def internal(*args, **kwargs):
    """ Print internal debug/warning message with given parameters. """

    if MIN_OUTPUT_IMPORTANCE.value > MessageImportance.INTERNAL.value:
        return

    print("\tINTERNAL: ", *args, file=sys.stderr, **kwargs)

def unchecked(name: str, *args, **kwargs):
    """ Print internal message informing the user, that a part of code was just run, which
        had not yet been tested. """
    
    if MIN_OUTPUT_IMPORTANCE.value > MessageImportance.UNCHECKED.value:
        return

    internal(f"This code has not yet been tested:'{name}'. If everything is working fine, please remove this message.")
    
def expectType(obj, expectedType, msg: str=""):
    if type(obj) != expectedType:
        warning(msg,":", f"Object '{obj}' is of type '{type(obj)}' where '{expectedType}' was expected!") 

def requireType(obj, requiredType, msg: str=""):
    if type(obj) != requiredType:
        error(Code.INVALID_TYPE, f"Object '{obj}' is of type '{type(obj)}' where '{requiredType}' was required!",msg) 

def expectEqual(val1, val2, msg: str=""):
    """ Compare two values. If they are different, print warning message. """

    try:
        if val1 != val2:
            warning(msg,":",f"'{val1}' does not equal '{val2}'!")
    except:
        warning(msg,":",f"Values '{val1}' and '{val2}' could not be compared!")

def expectEqualLists(list1, list2, msg: str=""):
    """ Compare two lists. If they are different, print warning message. """
    
    try:

        equal = True

        if len(list1) != len(list2):
            equal = False

        for el1, el2 in zip(list1,list2):
            if el1 != el2:
                equal = False
                break

        if not equal:
            warning(msg,":",f"'{list1}' does not equal '{list2}'!")

    except:
        warning(msg,":",f"Lists '{list1}' and '{list2}' could not be compared!")
