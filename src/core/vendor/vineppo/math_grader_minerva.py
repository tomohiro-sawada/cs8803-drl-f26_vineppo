# Vendored from McGill-NLP/VinePPO, commit 9f716e68bb934e71da86c226c1699c36b5553faf.
# MIT; see LICENSE.VinePPO and NOTICE.md for attribution and modifications.
# ruff: noqa
from copy import deepcopy
from math import isclose
from typing import Union
from tokenize import TokenError
from sympy.polys.polyerrors import PolynomialError

# Expected malformed/unsupported mathematical inputs, not arbitrary worker faults.
MATH_INPUT_ERRORS = (
    ValueError,
    TypeError,
    SyntaxError,
    TokenError,
    PolynomialError,
    NotImplementedError,
    ArithmeticError,
)
import regex
from sympy import simplify
from ..._safe_symbolic import parse_expr, numeric as N
from .math_grader import (
    _sympy_parse,
    should_allow_eval,
    _parse_latex,
    _inject_implicit_mixed_number,
)


def parse_digits(num):
    # format: 234.23 || 23%
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except ValueError:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except ValueError:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def symbolic_equal(a_latex, b_latex):
    def _parse(s, fast_latex=False):
        latex_parser = _parse_latex
        for f in [latex_parser, parse_expr]:
            try:
                return f(s)
            except MATH_INPUT_ERRORS:
                pass
        return s

    if not should_allow_eval(_parse(a_latex, fast_latex=True)) or not should_allow_eval(
        _parse(b_latex, fast_latex=True)
    ):
        return False

    a = _parse(a_latex, fast_latex=True)
    b = _parse(b_latex, fast_latex=True)

    try:
        expr = f"({a})-({b})"
        expr = _inject_implicit_mixed_number(expr)
        expr = expr.replace(" ", "")
        if simplify(_sympy_parse(expr)) == 0:
            return True
    except MATH_INPUT_ERRORS:
        pass

    try:
        if isclose(N(a), N(b), abs_tol=1e-3):
            return True
    except MATH_INPUT_ERRORS:
        pass
    return False


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    if str(prediction) == str(reference):
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if isclose(item, prediction, abs_tol=1e-3):
                            return True
                    else:
                        if item == prediction:
                            return True
                except (ValueError, ArithmeticError):
                    continue
            return False
    except (ValueError, ArithmeticError):
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True

    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close
        ):
            return True

    # symbolic equal with sympy
    # All calls run in a bounded worker (no nested process pool).
    if symbolic_equal(prediction, reference):
        return True

    return False


def is_correct(item, pred_key="prediction", prec=1e-3):
    pred = item[pred_key]
    ans = item["answer"]
    if isinstance(pred, list) and isinstance(ans, list):
        pred_matched = set()
        ans_matched = set()
        for i in range(len(pred)):
            for j in range(len(ans)):
                item_cpy = deepcopy(item)
                item_cpy.update({pred_key: pred[i], "answer": ans[j]})
                if is_correct(item_cpy, pred_key=pred_key, prec=prec):
                    pred_matched.add(i)
                    ans_matched.add(j)

        return len(pred_matched) == len(pred) and len(ans_matched) == len(ans)
    elif isinstance(pred, str) and isinstance(ans, str):
        if "\\cup" in pred and "\\cup" in ans:
            item = deepcopy(item)
            item.update(
                {
                    pred_key: pred.split("\\cup"),
                    "answer": ans.split("\\cup"),
                }
            )
            return is_correct(item, pred_key=pred_key, prec=prec)
        else:
            label = False
            try:
                label = (
                    abs(
                        float(regex.sub(r",", "", str(pred)))
                        - float(regex.sub(r",", "", str(ans)))
                    )
                    < prec
                )
            except (ValueError, ArithmeticError):
                pass
            label = label or (ans and pred == ans) or math_equal(pred, ans)
            return label
    else:
        raise NotImplementedError()


def eval_math(item, pred_key="prediction", prec=1e-3):
    pred = item[pred_key]
    if pred_key == "program_output" and isinstance(pred, str):
        pred = [pred]
    ans = item["answer"]
    if isinstance(pred, list) and isinstance(ans, list):
        # for some questions in MATH, `reference` repeats answers
        _ans = []
        for a in ans:
            if a not in _ans:
                _ans.append(a)
        ans = _ans
        # some predictions for MATH questions also repeats answers
        _pred = []
        for a in pred:
            if a not in _pred:
                _pred.append(a)
        # some predictions mistakenly box non-answer strings
        pred = _pred[-len(ans) :]

    item.update({pred_key: pred, "answer": ans})
    return is_correct(item, pred_key=pred_key, prec=prec)
