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

class WaveEqn:
    def __init__(self,
                 grid: np.array,               # grid
                 cs: float,                    # sound speed coefficient
                 bc_lower: str='constant:0',   # lower boundary condition
                 bc_upper: str='constant:0',   # upper boundary condition
                 ):
        """ 1D wave equation in cartesian coordinates """
        ## setting up the ghost grid
        self.npts = grid.size
        self.grid   = np.copy( grid )
        self.dr     = self.grid[1] - self.grid[0]
        self.ghostGrid       = np.zeros(self.npts + 2)
        self.ghostGrid[1:-1] = self.grid
        self.ghostGrid[0]    = grid[0] - self.dr
        self.ghostGrid[-1]   = grid[-1] + self.dr
        self.u_ghost         = np.zeros_like(self.ghostGrid)
        self.dudt_ghost      = np.zeros_like(self.ghostGrid)
        
        ## second order spatial difference
        self.ds  = (self.ghostGrid[2:] - self.ghostGrid[:-2]) / 2
        self.ds2 = self.ds**2
        
        ## generate boundary conditions
        self.cs   = cs
        self.bc_u = GenerateBCs1D(bc_lower, bc_upper)
        

    def __call__(self, t: np.array, y: np.array):
        """ Right hand side of the diffusion equation """
        u    = np.copy(y[:self.npts])
        dudt = np.copy(y[self.npts:])
        self.u_ghost[1:-1] = u

        ## apply boundary conditions
        self.bc_u(self.u_ghost, self.ghostGrid)
        
        d2u_dt2 = self.cs * (self.u_ghost[2:] - 2 * self.u_ghost[1:-1] + self.u_ghost[:-2]) / self.ds2
        return np.concatenate((dudt, d2u_dt2))
    

if __name__ == '__main__':
    #%% set up the grid
    tGrid = np.linspace(0, 3, num=30)
    rGrid  = np.linspace(0, 10, num=300)
    ## generate initial conditions
    u0    = np.zeros_like(rGrid)
    dudt0 = np.zeros_like(rGrid)
    u0[np.logical_and(rGrid > 4.5, rGrid < 5.5)] = 2
    y0 = np.concatenate((u0, dudt0))

    ## solve the initial value problem
    wave = WaveEqn(rGrid, 3.0,
                   bc_lower='constant:0',
                   bc_upper='constant:0'
                   )
    res = solve_ivp(wave, [tGrid.min(), tGrid.max()], y0, 
                    method='RK45', t_eval=tGrid)
    
    times = res.t 
    u_t   = res.y[:rGrid.size]
    dudt_t = res.y[rGrid.size:]
    
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