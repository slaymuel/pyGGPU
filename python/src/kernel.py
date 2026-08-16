import inspect
import ast
import sys
import pathlib
from enum import Enum
import functools
import irbuilder
from serialization.operands import Buffer, Const, Types
import numpy

root = pathlib.Path(__file__).resolve().parent
build_dir = root.parents[1] / "build"
sys.path.insert(0, str(build_dir))
import pyggpu

class Target(Enum):
    CPU = 1
    GPU = 2

class KernelArg:
    def __init__(self, name, value):
        self.name = name
        self.value = value

def kernel(blocks=None, threads=None):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Check if the function has already been compiled
            # If it has, launch the kernel with the provided arguments
            # If not, compile, cache and launch
            src = inspect.getsource(fn)
            tree = ast.parse(src)

            builder = irbuilder.IRBuilder()

            sig = inspect.signature(fn)
            params = sig.parameters
            inject_names(builder, params, args)

            builder.visit(tree)
            print("Env:")
            for name, value in builder.env.items():
                print(f"{name}: {value}")
            print("\nOps:")
            for op in builder.ops:
                print(op)
            print("\n")

            s_tree = ast.dump(tree, annotate_fields=True, include_attributes=True)

            engine = pyggpu.PyGGpu()
            #engine.compile(s_tree)
            ir = builder.serialize()
            #engine.compile(ir)

            wrapped_args = []
            for name, value in zip(params.keys(), args):
                wrapped_args.append(KernelArg(name, value))
            wrapped_args = tuple(wrapped_args)

            engine.launch(fn.__name__, pyggpu.KernelTarget.CPU, ir, wrapped_args, kwargs)
            # return engine.launch(fn.__name__, *args, **kwargs)
            return fn(*args, **kwargs)

        return wrapper
        
    def inject_names(builder, params, args):
        '''
        Injects the names of the parameters into the IRBuilder's environment.
        '''
        for (name, param), value in zip(params.items(), args):
            if isinstance(value, numpy.ndarray):
                builder.env[name] = Buffer(name, value.shape, value.dtype)
            else:
                builder.env[name] = Const(Types.INT, value=value)

    return decorator
