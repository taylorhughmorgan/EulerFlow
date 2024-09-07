# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:33:35 2024

@author: Hugh Morgan
"""
validBCs = ['reflective', 'gradient', 'constant', 'extrapolated']

def agnosticBC(ids, bc_id):
    """ Boundary condition agnostic to upper or lower bound """
    if bc_id == 'reflective':
        def applyBCs(u, grid):
            u[ids[0]]   = -u[ids[1]]
            
    elif 'gradient' in bc_id: 
        val = float( bc_id.split(':')[1] )
        def applyBCs(u, grid):
            dx = grid[ids[1]] - grid[ids[0]]
            u[ids[0]]   = u[ids[1]] + val * dx
            
    elif bc_id == 'extrapolated':
        def applyBCs(u, grid):
            u[ids[0]]   = 2 * u[ids[1]] - u[ids[2]]
            
    elif 'constant' in bc_id:
        val = float(bc_id.split(':')[1])
        def applyBCs(u, grid):
            u[ids[0]] = val
    else:
        raise Exception(f"Boundary condition '{bc_id}' has not been implemented. Valid options are {validBCs}.")
    
    return applyBCs
    

def GenerateBCs1D(bc_lower: str, bc_upper: str):
    """ Apply Boundary conditions. """
    ## boundary conditions on the lower bound
    id_lower = [0, 1, 2]
    applyLowerBCs = agnosticBC(id_lower, bc_lower)
    
    ## boundary conditions on the upper bound
    id_upper = [-1, -2, -3]
    applyUpperBCs = agnosticBC(id_upper, bc_upper)
    
    ## combine them into a single boundary condition
    def applyBCs(u, grid):
        applyLowerBCs(u, grid)
        applyUpperBCs(u, grid)
        
    return applyBCs