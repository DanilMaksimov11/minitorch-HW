from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """
    `ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes:
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


class Scalar:
    """
    A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    history: Optional[ScalarHistory]
    derivative: Optional[float]
    data: float
    unique_id: int
    name: str

    def __init__(
        self,
        v: float,
        back: ScalarHistory = ScalarHistory(),
        name: Optional[str] = None,
    ):
        global _var_count
        _var_count += 1
        self.unique_id = _var_count
        self.data = float(v)
        self.history = back
        self.derivative = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

    def __repr__(self) -> str:
        """Return string representation of the Scalar object.
        
        Returns:
            String representation showing the scalar value
        """
        return "Scalar(%f)" % self.data

    def __mul__(self, b: ScalarLike) -> Scalar:
        """Multiply this scalar by another scalar-like object.
        
        Args:
            b: Scalar-like object to multiply with
            
        Returns:
            New Scalar result of multiplication operation
        """
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        """Divide this scalar by another scalar-like object.
        
        Args:
            b: Scalar-like object to divide by
            
        Returns:
            New Scalar result of division operation
        """
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        """Reverse division: divide another scalar-like object by this scalar.
        
        Args:
            b: Scalar-like object to be divided
            
        Returns:
            New Scalar result of reverse division operation
        """
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: ScalarLike) -> Scalar:
        """Add another scalar-like object to this scalar.
        
        Args:
            b: Scalar-like object to add
            
        Returns:
            New Scalar result of addition operation
        """
        return Add.apply(self, b)

    def __bool__(self) -> bool:
        """Convert scalar to boolean based on its value.
        
        Returns:
            True if scalar value is non-zero, False otherwise
        """
        return bool(self.data)

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Compare if this scalar is less than another scalar-like object.
        
        Args:
            b: Scalar-like object to compare against
            
        Returns:
            New Scalar result of less-than comparison (1.0 if true, 0.0 if false)
        """
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Compare if this scalar is greater than another scalar-like object.
        
        Args:
            b: Scalar-like object to compare against
            
        Returns:
            New Scalar result of greater-than comparison (1.0 if true, 0.0 if false)
        """
        return LT.apply(b, self)

    def __eq__(self, b: ScalarLike) -> Scalar:  # type: ignore[override]
        """Compare if this scalar is equal to another scalar-like object.
        
        Args:
            b: Scalar-like object to compare against
            
        Returns:
            New Scalar result of equality comparison (1.0 if true, 0.0 if false)
        """
        return EQ.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Subtract another scalar-like object from this scalar.
        
        Args:
            b: Scalar-like object to subtract
            
        Returns:
            New Scalar result of subtraction operation
        """
        return self + (-b)

    def __neg__(self) -> Scalar:
        """Negate this scalar (multiply by -1).
        
        Returns:
            New Scalar with negated value
        """
        return Neg.apply(self)

    def __radd__(self, b: ScalarLike) -> Scalar:
        """Reverse addition: add this scalar to another scalar-like object.
        
        Args:
            b: Scalar-like object to add to
            
        Returns:
            New Scalar result of reverse addition operation
        """
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        """Reverse multiplication: multiply another scalar-like object by this scalar.
        
        Args:
            b: Scalar-like object to multiply
            
        Returns:
            New Scalar result of reverse multiplication operation
        """
        return self * b

    def log(self) -> Scalar:
        """Compute natural logarithm of this scalar.
        
        Returns:
            New Scalar result of natural logarithm operation
        """
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Compute exponential of this scalar.
        
        Returns:
            New Scalar result of exponential operation
        """
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Compute sigmoid function of this scalar.
        
        Returns:
            New Scalar result of sigmoid operation
        """
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Compute ReLU (Rectified Linear Unit) of this scalar (max(0, value)).
        
        Returns:
            New Scalar result of ReLU operation
        """
        return ReLU.apply(self)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """
        Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            x: value to be accumulated
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        "True if this variable created by the user (no `last_fn`)"
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:

        """Applies the chain rule for backpropagation through the last executed function.

        This function performs the backward pass by computing local derivatives of the
        last operation with respect to its inputs, then pairs these derivatives with
        their corresponding input variables. 

        Args:
            d_output: The gradient of the loss with respect to the output of this
                variable. This serves as the upstream gradient for chain rule
                computation.

        Returns:
            An iterable of tuples where each tuple contains:
                - A Variable that was used as input to the forward function
                - The corresponding derivative of the loss with respect to that Variable
        """

        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        if type(h.last_fn.backward(d_output = d_output, ctx = h.ctx)) is float:
            return zip(h.inputs, [h.last_fn.backward(d_output = d_output, ctx = h.ctx)])
        else:
            return zip(h.inputs, h.last_fn.backward(d_output = d_output, ctx = h.ctx))

    def backward(self, d_output: Optional[float] = None) -> None:
        """
        Calls autodiff to fill in the derivatives for the history of this object.

        Args:
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """
    Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Parameters:
        f : function from n-scalars to 1-scalar.
        *scalars  : n input scalar values.
    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
