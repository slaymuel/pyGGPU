from serialization import ir_pb2


def ssa_name(value):
    """Return the SSA string name expected by proto string fields."""
    return value.name if hasattr(value, "name") else value

class AddOp:
    kind = "add"
    def __init__(self, result, lhs, rhs):
        self.result = result
        self.lhs = lhs
        self.rhs = rhs

    def to_proto(self):
        msg = ir_pb2.AddOp()
        msg.result = ssa_name(self.result)
        msg.lhs.CopyFrom(self.lhs.to_proto())
        msg.rhs.CopyFrom(self.rhs.to_proto())
        return msg

    def __str__(self):
        return f"{self.result} = add {self.lhs}, {self.rhs}"

class LoadOp:
    kind = "load"
    def __init__(self, result, buffer, index):
        self.result = result
        self.buffer = buffer
        self.index = index

    def to_proto(self):
        msg = ir_pb2.LoadOp()
        msg.result = ssa_name(self.result)
        msg.buffer = self.buffer.name
        msg.index.CopyFrom(self.index.to_proto())
        return msg

    def __str__(self):
        return f"{self.result} = load {self.buffer}[{self.index}]"

class StoreOp:
    kind = "store"
    def __init__(self, value, buffer, index):
        self.value = value
        self.buffer = buffer
        self.index = index

    def to_proto(self):
        msg = ir_pb2.StoreOp()
        msg.value.CopyFrom(self.value.to_proto())
        msg.buffer = self.buffer.name
        msg.index.CopyFrom(self.index.to_proto())
        return msg

    def __str__(self):
        return f"{self.buffer}[{self.index}] = store {self.value}"

class ConstOp:
    kind = "constant"
    def __init__(self, value, result):
        self.value = value
        self.result = result

    def to_proto(self):
        msg = ir_pb2.ConstOp()
        print(self.value)
        msg.value.CopyFrom(self.value.to_proto())
        msg.result = ssa_name(self.result)
        return msg
        # msg = ir_pb2.Operand()
        # msg.int_const = self.value
        # return msg

    def __str__(self):
        return str(self.value)