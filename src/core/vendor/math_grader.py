# Vendored from McGill-NLP/VinePPO, commit 9f716e68bb934e71da86c226c1699c36b5553faf.
# MIT; see LICENSE.VinePPO and NOTICE.md for attribution and modifications.
# ruff: noqa
import re
from pylatexenc import latex2text
from .safe_symbolic import parse_expr

def _sympy_parse(expr):
    return parse_expr(expr, implicit=True)

def _latex_to_text(expr):
    # The enclosing worker bounds the entire operation, not just a thread wait.
    return latex2text.LatexNodes2Text().latex_to_text(expr)

BAD_SUBSTRINGS = ["^{", "^("]

BAD_REGEXES = [
    "\^[0-9]+\^",
    "\^-?[0-9][0-9]+",
    "\*\*[0-9]+\*\*",
    "\*\*-?[0-9][0-9]+",
    "[0-9][0-9][0-9]+!",
]

def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = _latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()

def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step

def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    expr = expr.replace("pi", "")
    expr = expr.replace("inf", "")
    expr = expr.replace("cos", "")
    expr = expr.replace("sin", "")
    expr = expr.replace("tan", "")
    expr = expr.replace("cot", "")
    expr = expr.replace("arccos", "")
    expr = expr.replace("arcsin", "")
    expr = expr.replace("arctan", "")
    expr = expr.replace("arccot", "")
    expr = expr.replace("log", "")
    expr = expr.replace("ln", "")
    expr = expr.replace("exp", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)

def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True
