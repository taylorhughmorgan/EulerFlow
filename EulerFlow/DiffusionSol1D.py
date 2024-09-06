#!/usr/bin/env python
"""
Author: Hugh Morgan
Date: 2024-09-05
Description: Solve the 1D diffusion equation in cartesian, cylindrical, and spherical coordinates.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from BoundaryConditions import GenerateBCs1D

class Diffusion:
    def __init__(self,
                 grid: np.array,                # grid
                 diffCoef: float,               # diffusion coefficient
                 bc_lower: str='constant:2',  # lower boundary condition
                 bc_upper: str='constant:1',    # upper boundary condition
                 ):
        """ 1D diffusion equation in cartesian coordinates """
        ## setting up the ghost grid
        self.npts = grid.size
        self.grid   = np.copy( grid )
        self.dr     = self.grid[1] - self.grid[0]
        self.ghostGrid       = np.zeros(self.npts + 2)
        self.ghostGrid[1:-1] = self.grid
        self.ghostGrid[0]    = grid[0] - self.dr
        self.ghostGrid[-1]   = grid[-1] + self.dr
        
        ## second order spatial difference
        self.ds  = (self.ghostGrid[2:] - self.ghostGrid[:-2]) / 2
        self.ds2 = self.ds**2
        
        ## generate boundary conditions
        self.diffCoef = diffCoef
        self.bc_lower, self.bc_upper = GenerateBCs1D(bc_lower, bc_upper)
        

    def __call__(self, t: np.array, u: np.array):
        """ Right hand side of the diffusion equation """
        u_ghost = np.zeros(self.npts + 2)
        u_ghost[1:-1] = np.copy(u)

        ## apply boundary conditions
        self.bc_lower(u_ghost)
        self.bc_upper(u_ghost)
        
        du_dt = self.diffCoef * (u_ghost[2:] - 2 * u_ghost[1:-1] + u_ghost[:-2]) / self.ds2
        return du_dt
    

if __name__ == '__main__':
    #%% set up the grid
    tGrid = np.linspace(0, 3, num=100)
    rGrid  = np.linspace(0, 10, num=300)
    ## generate initial conditions
    u0 = np.ones_like(rGrid)
    u0[np.logical_and(rGrid > 3, rGrid < 6)] = 2

    ## solve the initial value problem
    diffusion = Diffusion(rGrid, 1.0,
                          bc_lower='constant:1',
                          bc_upper='transmissive'
                          )
    res = solve_ivp(diffusion, [tGrid.min(), tGrid.max()], u0, 
                    method='RK45', t_eval=tGrid)
    
    times = res.t 
    u_t   = res.y
    
    #%% plotting at discrete times
    fig, ax = plt.subplots()
    for it, tTemp in enumerate(times):
        ax.plot(rGrid, u_t[:,it], label=f"time={tTemp:.3f}s")
    ax.grid(True)
    ax.legend()
    
    #%% density plots
    extent = [tGrid.min(), tGrid.max(), rGrid.min(), rGrid.max()]
    fig, ax = plt.subplots()
    
    cset = ax.pcolormesh(tGrid, rGrid, u_t)
    ax.set_title("Temperature vs distance and time")
    fig.colorbar(cset, ax=ax)
    ax.set_ylabel("distance (m)")
    ax.set_xlabel("Time (s)")