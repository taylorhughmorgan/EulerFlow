# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:33:35 2024

@author: Hugh Morgan
"""


def GenerateBCs1D(bc_lower: str, bc_upper: str):
    """ Apply Boundary conditions. """
    ## boundary conditions on the lower bound
    if bc_lower == 'reflective':
        def applyLowerBCs(u):
            u[0]   = -u[1]
    elif bc_lower == 'transmissive':
        def applyLowerBCs(u):
            u[0]   = u[1]
    elif bc_lower == 'extrapolated':
        def applyLowerBCs(u):
            u[0]   = 2 * u[1] - u[2]
    elif 'constant' in bc_lower:
        val = float(bc_lower.split(':')[1])
        def applyLowerBCs(u):
            u[0] = val
    else:
        raise Exception(f"Boundary condition '{bc_lower}' has not been implemented.")
    
    ## boundary conditions on the upper bound
    if bc_upper == 'reflective':
        def applyUpperBCs(u):
            u[-1]   = -u[-2]
    elif bc_upper == 'transmissive':
        def applyUpperBCs(u):
            u[-1]   = u[-2]
    elif bc_upper == 'extrapolated':
        def applyUpperBCs(u):
            u[-1]   = 2 * u[-2] - u[-3]
    elif 'constant' in bc_upper:
        val = float(bc_upper.split(':')[1])
        def applyUpperBCs(u):
            u[-1] = val
    else:
        raise Exception(f"Boundary condition '{bc_upper}' has not been implemented.")

    return applyLowerBCs, applyUpperBCs