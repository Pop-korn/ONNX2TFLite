import sys
from enum import Enum

class Code(Enum):
    INPUT_FILE_ERR = 1
    UNSUPPORTED_OPERATOR = 2
    UNSUPPORTED_ONNX_TYPE = 3
    INVALID_TYPE = 4
    INVALID_TENSOR_SHAPE = 5

def error(err_code, *args, **kwargs):
    """ Print error message with given parameters and exit execution with given exit code. """
    print("\tERROR: ", *args, file=sys.stderr, **kwargs)
    exit(err_code)

def warning(*args, **kwargs):
    """ Print warning message with given parameters. """
    print("\tWARNING: ", *args, file=sys.stderr, **kwargs)

def internal(*args, **kwargs):
    """ Print internal debug/warning message with given parameters. """
    print("\tINTERNAL: ", *args, file=sys.stderr, **kwargs)

def unchecked(name: str, *args, **kwargs):
    """ Print internal message informing the user, that a part of code was just run, which
        had not yet been tested. """
    internal(f"This code has not yet been tested:'{name}'. If everything is working fine, please remove this message.")
    
def expectType(obj, expectedType, msg: str=""):
    if type(obj) != expectedType:
        warning(msg,":", f"Object '{obj}' is of type '{type(obj)}' where '{expectedType}' was expected!") 

def requireType(obj, requiredType, msg: str=""):
    if type(obj) != requiredType:
        error(Code.INVALID_TYPE, f"Object '{obj}' is of type '{type(obj)}' where '{requiredType}' was required!",msg) 
