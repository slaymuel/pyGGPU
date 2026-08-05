from serialization import ir_pb2
import enum

class Buffer:
    kind = "buffer"
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def to_proto(self):
        msg = ir_pb2.Operand()
        msg.buffer.name = self.name
        msg.buffer.shape.extend(self.shape)
        msg.buffer.dtype = str(self.dtype)
        return msg

class Types(enum.Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"

class Const:
    kind = "const"
    def __init__(self, kind, value):
        self.kind = kind
        self.value = value

    def to_proto(self):
        msg = ir_pb2.Operand()

        if self.kind == Types.INT:
            msg.int_const = self.value

        elif self.kind == Types.FLOAT:
            msg.float_const = self.value

        elif self.kind == Types.BOOL:
            msg.bool_const = self.value

        return msg

class SSA:
    kind = "ssa"
    def __init__(self, name):
        self.name = name

    def to_proto(self):
        msg = ir_pb2.Operand()
        msg.ssa = self.name
        return msg
        