import sys
from enum import Enum

class Code(Enum):
    INPUT_FILE_ERR = 1

def eprint(err_code, *args, **kwargs):
    """ Print error message with given parameters and exit execution with given exit code. """
    print("\tERROR: ", *args, file=sys.stderr, **kwargs)
    exit(err_code)

def wprint(*args, **kwargs):
    """ Print warning message with given parameters. """
    print("\tWARNING: ", *args, file=sys.stderr, **kwargs)
    