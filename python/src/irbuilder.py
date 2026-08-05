import ast
from serialization.ops import AddOp, LoadOp, StoreOp, ConstOp
from serialization.operands import SSA, Const, Types
import serialization.ir_pb2 as ir_pb2

class IRBuilder(ast.NodeVisitor):
    def __init__(self):
        # Arguments
        self.env = {}
        # Operations
        self.ops = []
        # Local variables
        self.locals = set()

        self.ssa_counter = 0

    def visit_Module(self, node):
        for stmt in node.body:
            self.visit(stmt)

    def visit_Assign(self, node):
        target = node.targets[0]
        value = self.visit(node.value)

        if isinstance(target, ast.Name):
            self.ops.append(ConstOp(value, target.id))
            self.locals.add(target.id)

        elif isinstance(target, ast.Subscript):
            base = self.visit(target.value)
            index = self.visit(target.slice)
            self.ops.append(StoreOp(value, base, index))

    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return Const(Types.INT, value=node.value)
        elif isinstance(node.value, float):
            return Const(Types.FLOAT, value=node.value)
        elif isinstance(node.value, bool):
            return Const(Types.BOOL, value=node.value)
        else:
            raise NotImplementedError("Unsupported constant type")

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)

        tmp = self.new_ssa()
        self.ops.append(AddOp(tmp, lhs, rhs))
        self.locals.add(tmp.name)
        return tmp

    def visit_Name(self, node):
        name = node.id

        # Kernel argument?
        if name in self.env:
            return self.env[name]

        # Local SSA value?
        if name in self.locals:
            return SSA(name)

        raise KeyError(f"Unknown variable {name}")


    def visit_FunctionDef(self, node):
        # Only visit the body, ignore the function name and args for now
        for stmt in node.body:
            self.visit(stmt)

    def visit_Subscript(self, node):
        base = self.visit(node.value)
        index = self.visit(node.slice)
        tmp = self.new_ssa()
        self.ops.append(LoadOp(tmp, base, index))
        return tmp

    def new_ssa(self):
        name = f"%t{self.ssa_counter}"
        self.ssa_counter += 1
        return SSA(name)

    def serialize(self):
        ir_msg = ir_pb2.IR()
        for op in self.ops:
            op_msg = ir_msg.ops.add()
            field = op.kind
            getattr(op_msg, field).CopyFrom(op.to_proto())

        # Also serialize the environment
        for name, operand in self.env.items():
            binding = ir_msg.env.bindings.add()
            binding.name = name
            binding.value.CopyFrom(operand.to_proto())


        return ir_msg
