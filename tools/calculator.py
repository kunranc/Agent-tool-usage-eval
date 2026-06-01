"""
Calculator tool — evaluates a single arithmetic expression.

Stateless. Safe: uses Python's `ast` to parse and walk only an explicit
whitelist of operators. Will NOT execute arbitrary code (no `eval()`).
"""

import ast
import operator

# Whitelist of allowed AST operator nodes -> Python operator functions.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_eval(node.operand))
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def calculator(expression: str) -> dict:
    """Evaluate `expression` and return {ok, result|error}."""
    try:
        tree = ast.parse(expression, mode="eval")
        return {"ok": True, "result": _eval(tree.body)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# Tool spec sent to the LLM. The description is the leverage point for
# tool selection — keep it specific about WHAT it does and what NOT to use
# it for.
SPEC = {
    "name": "calculator",
    "description": (
        "Evaluate a single arithmetic expression and return the result. "
        "Supports +, -, *, /, //, %, ** (exponent), parentheses, and unary minus. "
        "Use for any numerical math: addition, subtraction, multiplication, "
        "division, powers, percentages computed as fractions. "
        "Do NOT use for: unit conversion, symbolic algebra, equations with "
        "variables, statistics requiring data, or anything that requires "
        "external information. Returns a single number."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "Arithmetic expression to evaluate, e.g. '2 + 3 * (4 - 1)' "
                    "or '(100 - 17) / 4'."
                ),
            }
        },
        "required": ["expression"],
    },
}
