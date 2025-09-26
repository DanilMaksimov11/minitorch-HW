import random
import math
from typing import List, Tuple
from dataclasses import dataclass


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generates N random points in the unit square [0,1) x [0,1).

    Args:
        N: Number of random points to generate.

    Returns:
        List of N tuples, each containing (x1, x2) coordinates.
    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """A dataset containing points with binary labels.

    Attributes:
        N: Number of points in the dataset.
        X: List of (x1, x2) coordinate tuples for each point.
        y: List of binary labels (0 or 1) for each point.
    """
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generates a simple vertical split dataset.

    Points are labeled based on whether their x1 coordinate is less than 0.5.

    Args:
        N: Number of points to generate.

    Returns:
        Graph object with points and labels.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generates a dataset split by a diagonal line.

    Points are labeled based on whether x1 + x2 < 0.5.

    Args:
        N: Number of points to generate.

    Returns:
        Graph object with points and labels.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generates a dataset with two vertical splits.

    Points are labeled 1 if x1 < 0.2 or x1 > 0.8, otherwise labeled 0.

    Args:
        N: Number of points to generate.

    Returns:
        Graph object with points and labels.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generates an XOR-style dataset.

    Points are labeled 1 if they are in the upper-left or lower-right quadrants
    (relative to center 0.5, 0.5), otherwise labeled 0.

    Args:
        N: Number of points to generate.

    Returns:
        Graph object with points and labels.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generates a circular classification dataset.

    Points are labeled based on whether they fall outside a circle centered
    at (0.5, 0.5) with radius sqrt(0.1).

    Args:
        N: Number of points to generate.

    Returns:
        Graph object with points and labels.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates a two-spiral dataset.

    Creates two interlocking spirals with points labeled by spiral membership.

    Args:
        N: Number of points to generate (must be even).

    Returns:
        Graph object with spiral points and labels.
    """
    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}