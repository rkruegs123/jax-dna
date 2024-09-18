"""Symbolic base functions for DNA2 energy model.

This function is based on the oxDNA2 model paper found here:
https://ora.ox.ac.uk/objects/uuid:241ae8d5-2092-4b24-b1d0-3fb7482b7bcd/files/m7422ee58d9747bbd7af00d6435b570e6
page 16, eq A5
"""


def f6(theta: float, a: float, b: float) -> float:
    """This is a symbolic representation of the f6 base function.

    Equation A5 from the oxDNA2 paper.

    This function has described has 1 case:
    1. theta >= b
    2. Otherwise
    """
    value = 0
    if theta >= b:
        value = a / 2 * (theta - b) ** 2

    return value
