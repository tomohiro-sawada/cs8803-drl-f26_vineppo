"""Restricted SymPy construction; never eval/sympify a source string.

Keep upstream token transformations, but interpret only arithmetic AST nodes and
an explicit set of mathematical constructors. This is a security restriction,
not an alternative equality rule. Run only inside the resource-bounded worker.
"""

import ast
import operator
import re

import sympy
from sympy.parsing import sympy_parser

_FUNCTIONS = {
    name: getattr(sympy, name)
    for name in (
        "sqrt",
        "sin",
        "cos",
        "tan",
        "cot",
        "asin",
        "acos",
        "atan",
        "acot",
        "sinh",
        "cosh",
        "tanh",
        "log",
        "exp",
        "Abs",
        "factorial",
    )
}
_CONSTANTS = {"pi": sympy.pi, "E": sympy.E, "I": sympy.I, "oo": sympy.oo}
_CONSTRUCTORS = {
    name: getattr(sympy, name) for name in ("Symbol", "Integer", "Float", "Rational")
}
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _construct(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return sympy.Number(node.value)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_construct(node.left), _construct(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_construct(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and not node.keywords
    ):
        name = node.func.id
        if name in _CONSTRUCTORS:
            # Strings are permitted only as constructor literals created by
            # SymPy's transformations, never as expressions for sympification.
            values = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
            if len(values) != len(node.args) or not values:
                raise ValueError("Invalid constructor arguments")
            pattern = (
                r"[A-Za-z][A-Za-z0-9]*"
                if name == "Symbol"
                else r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
            )
            if not all(
                type(v) in (str, int, float) and re.fullmatch(pattern, str(v))
                for v in values
            ):
                raise ValueError("Invalid constructor literal")
            return _CONSTRUCTORS[name](*values)
        if name in _FUNCTIONS:
            return _FUNCTIONS[name](*map(_construct, node.args))
    raise ValueError("Unsupported symbolic syntax")


def parse_expr(source: str, *, implicit: bool = False):
    if not re.fullmatch(r"[A-Za-z0-9+*/^().,!\s-]+", source):
        raise ValueError("Unsupported symbolic characters")
    depth = 0
    for char in source:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise SyntaxError(
                    "Unmatched closing parenthesis in symbolic expression"
                )
    if depth:
        raise SyntaxError("Unmatched opening parenthesis in symbolic expression")
    transforms = sympy_parser.standard_transformations
    if implicit:
        source = source.replace("^", "**")
        transforms += (sympy_parser.implicit_multiplication_application,)
    namespace = _FUNCTIONS | _CONSTANTS | _CONSTRUCTORS
    code = sympy_parser.stringify_expr(source.strip(), {}, namespace, transforms)
    return _construct(ast.parse(code, mode="eval").body)


def numeric(value):
    return sympy.N(parse_expr(value) if isinstance(value, str) else value)
