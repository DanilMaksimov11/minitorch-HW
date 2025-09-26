from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """

    lhs_vals = list(vals)
    lhs_vals[arg] += epsilon

    rhs_vals = list(vals)
    rhs_vals[arg] -= epsilon

    return (f(*lhs_vals) - f(*rhs_vals))/(2*epsilon)

variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """

    topological_list = []
    ids_list = []
    
    def visit(var: Variable):
        for parent in var.parents:
            if parent.unique_id not in ids_list:
                visit(parent)
        topological_list.append(var)
        ids_list.append(var.unique_id)

    visit(variable)
    return topological_list[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """

    var_deriv_dict = {variable.unique_id: deriv}
    topological_list = topological_sort(variable)

    for node in topological_list:
        if node.is_leaf():
            node.accumulate_derivative(var_deriv_dict[node.unique_id])
        else:
            for parent, derivative in node.chain_rule(d_output = var_deriv_dict[node.unique_id]):
                if parent.unique_id not in var_deriv_dict:
                    var_deriv_dict[parent.unique_id] = derivative
                else:
                    var_deriv_dict[parent.unique_id] += derivative

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
