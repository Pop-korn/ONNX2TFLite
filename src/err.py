import sys

def eprint(*args, **kwargs):
    print("ERROR: ", *args, file=sys.stderr, **kwargs)

def wprint(*args, **kwargs):
    print("WARNING: ", *args, file=sys.stderr, **kwargs)