import flatbuffers as fb

import generator.model.OperatorCodes as oc
import generator.model.SubGraphs as sg

class Model:

    def __init__(self, version: int, description: str, operatorCodes: oc.OperatorCodes
                , subgraphs: sg.SubGraphs) -> None:
        pass
