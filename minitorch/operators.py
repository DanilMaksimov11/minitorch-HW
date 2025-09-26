"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable
# Implementation of a prelude of elementary functions.

# Mathematical functions:

# - mul
def mul(x: float, y: float) -> float:
    """
    Multiplies two numbers and returns the result.
    
    Args:
        x: First operand for multiplication
        y: Second operand for multiplication
    
    Returns:
        The product of x and y (x * y)
    """
    return x*y

# - id
def id(x: float) -> float:
    """
    Identity function - returns the input value unchanged.
    
    Args:
        x: Input value
        
    Returns:
        The same value x (no transformation applied)
    """
    return x

# - add
def add(x: float, y: float) -> float:
    """
    Adds two numbers and returns their sum.
    
    Args:
        x: First operand for addition
        y: Second operand for addition
    
    Returns:
        The sum of x and y (x + y)
    """
    return x+y

# - neg
def neg(x: float) -> float:
    """
    Negates the input value (multiplies by -1).
    
    Args:
        x: Input value to be negated
        
    Returns:
        The negative of x (-x)
    """
    return x * (-1)

# - lt
def lt(x: float, y: float) -> bool:
    """
    Checks if x is less than y.
    
    Args:
        x: Left operand for comparison
        y: Right operand for comparison
    
    Returns:
        True if x < y, False otherwise
    """
    return x < y

# - eq
def eq(x: float, y: float) -> bool:
    """
    Checks if x is equal to y.
    
    Args:
        x: First value for equality comparison
        y: Second value for equality comparison
    
    Returns:
        True if x equals y, False otherwise
    """
    return x == y 

# - max
def max(x: float, y: float) -> float:
    """
    Returns the maximum of two numbers.
    
    Args:
        x: First value to compare
        y: Second value to compare
    
    Returns:
        The larger value between x and y
    """
    if x > y:
        return x
    else:
        return y

# - is_close
def is_close(x: float, y: float) -> bool:
    """
    Checks if two numbers are close within a tolerance of 1e-2.
    Useful for floating-point comparisons to avoid precision issues.
    
    Args:
        x: First value to compare
        y: Second value to compare
    
    Returns:
        True if |x - y| < 0.01, False otherwise
    """
    return math.fabs(x - y) < 1e-2

# - sigmoid
def sigmoid(x: float) -> float:
    """
    Computes the sigmoid (logistic) function of x: 1/(1 + e^(-x)).
    Uses numerically stable implementation for negative values.
    
    Args:
        x: Input value to the sigmoid function
        
    Returns:
        Sigmoid value between 0 and 1
    """
    if x >= 0:
        return 1/(1 + math.exp(-x))
    else:
        return math.exp(x)/(1 + math.exp(x))
    
# - relu
def relu(x: float) -> float:
    """
    Computes the Rectified Linear Unit (ReLU) function: max(0, x).
    
    Args:
        x: Input value to the ReLU function
        
    Returns:
        x if x >= 0, 0 otherwise
    """
    if x >= 0:
        return x
    else:
        return 0

# - log
def log(x: float) -> float:
    """
    Computes the natural logarithm (base e) of x.
    
    Args:
        x: Input value (must be positive)
        
    Returns:
        Natural logarithm of x
    """
    return math.log(x) 

# - exp
def exp(x: float) -> float:
    """
    Computes the exponential function e^x.
    
    Args:
        x: Exponent value
        
    Returns:
        e raised to the power x
    """
    return math.exp(x)

# - log_back
def log_back(x: float, dL_dy: float) -> float:
    """
    Computes the gradient of the logarithm function for backpropagation.
    Derivative of log(x) is 1/x, multiplied by upstream gradient dL_dy.
    
    Args:
        x: Input value at which derivative is evaluated
        dL_dy: Upstream gradient from subsequent layers
        
    Returns:
        Gradient of loss with respect to x: dL/dx = dL/dy * dy/dx = dL/dy * (1/x)
    """
    return dL_dy/x

# - inv
def inv(x: float) -> float:
    """
    Computes the multiplicative inverse (reciprocal) of x: 1/x.
    
    Args:
        x: Input value (cannot be zero)
        
    Returns:
        Reciprocal of x (1/x)
    """
    return 1/x

# - inv_back
def inv_back(x: float, dL_dy: float) -> float:
    """
    Computes the gradient of the inverse function for backpropagation.
    Derivative of 1/x is -1/x², multiplied by upstream gradient dL_dy.
    
    Args:
        x: Input value at which derivative is evaluated
        dL_dy: Upstream gradient from subsequent layers
        
    Returns:
        Gradient of loss with respect to x: dL/dx = dL/dy * dy/dx = dL/dy * (-1/x²)
    """
    return - dL_dy / x**2

# - relu_back
def relu_back(x: float, dL_dy: float) -> float:
    """
    Computes the gradient of the ReLU function for backpropagation.
    Derivative is 1 for x > 0, 0 for x <= 0, multiplied by upstream gradient dL_dy.
    
    Args:
        x: Input value at which derivative is evaluated
        dL_dy: Upstream gradient from subsequent layers
        
    Returns:
        Gradient of loss with respect to x: dL/dx = dL/dy * dy/dx
        where dy/dx = 1 if x > 0, 0 otherwise
    """
    return float(x > 0) * dL_dy

#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

def map(f: Callable, array: Iterable) -> Iterable:
    """Applies a function to each element in an iterable.
    
    Args:
        f: A callable that takes one argument and returns a value.
        array: An iterable of elements to be processed.
        
    Yields:
        The result of applying f to each element of the input iterable.
    """
    for x in array:
        yield f(x)


def zipWith(f: Callable, array_x: Iterable, array_y: Iterable) -> Iterable:
    """Combines two iterables element-wise using a given function.
    
    Args:
        f: A callable that takes two arguments and returns a value.
        array_x: First iterable for combination.
        array_y: Second iterable for combination.
        
    Yields:
        The result of applying f to each pair (x, y) from the input iterables.
        Stops when the shorter iterable is exhausted.
    """
    for x, y in zip(array_x, array_y):
        yield f(x, y)


def reduce(f: Callable, array: Iterable) -> Any:
    """Reduces an iterable to a single value using a binary function.
    
    Applies the function cumulatively from left to right. For example:
    reduce(f, [a, b, c, d]) = f(f(f(a, b), c), d)
    
    Args:
        f: A callable that takes two arguments and returns a value.
        array: An iterable to be reduced. Must be non-empty unless handled by caller.
        
    Returns:
        The final accumulated value.
        
    Raises:
        StopIteration: If the input iterable is empty.
    """
    it = iter(array)
    value = next(it)
    for element in it:
        value = f(value, element)
    return value


# Use these to implement

def negList(array: Iterable) -> Iterable:
    """Negates each element in the input iterable.
    
    Args:
        array: An iterable of numbers.
        
    Returns:
        An iterable where each element is negated.
    """
    return map(neg, array)


def addLists(array_x: Iterable, array_y: Iterable) -> Iterable:
    """Adds elements from two iterables pairwise.
    
    Args:
        array_x: First iterable of numbers.
        array_y: Second iterable of numbers.
        
    Returns:
        An iterable containing pairwise sums.
        Stops when the shorter input iterable is exhausted.
    """
    return zipWith(add, array_x, array_y)


def sum(array: Iterable) -> Any:
    """Computes the sum of all elements in the iterable.
    
    Args:
        array: An iterable of numbers. Can be empty.
        
    Returns:
        The sum of all elements. Returns 0 for empty input.
    """
    if array:
        return reduce(add, array)
    else:
        return 0


def prod(array: Iterable) -> Any:
    """Computes the product of all elements in the iterable.
    
    Args:
        array: An iterable of numbers. Can be empty.
        
    Returns:
        The product of all elements. Returns 1 for empty input.
    """
    if array:
        return reduce(mul, array)
    else:
        return 1