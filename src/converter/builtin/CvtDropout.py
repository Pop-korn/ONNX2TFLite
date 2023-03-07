import src.parser.builtin.Dropout as onnxDropout

import src.generator.meta.meta as tflMeta

def convert(oDropout: onnxDropout.Dropout) -> tflMeta.BuiltinOptions | None:
    """ Convert the ONNX 'Dropout' operator to TFLite. 
        There is no direct equivalent.
        
        May return 'None' to indicate that the operator is to be skipped!
    """
    return None