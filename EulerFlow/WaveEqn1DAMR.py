#!/usr/bin/env python
"""
Author: Hugh Morgan
Date: 2024-09-05
Description: Solve the 1D diffusion equation in cartesian, cylindrical, and spherical coordinates.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from EulerFlow.BoundaryConditions import GenerateBCs1D

class AdaptiveGrid:
    """ interpolating onto adaptive grid """
    def __init__(self, 
                 grid: np.array, 
                 field: np.array, 
                 epsilon: float=1.0, 
                 minval: float=1e-4,
                 smooth: bool=True,
                 ):
        ## decomposing and mapping onto uniform grid
        self.grid = grid
        # normalize r
        gridNorm = (grid - grid.min()) / (grid.max() - grid.min())

        # compute gradient magnitude
        grad = np.abs(np.gradient(field, gridNorm) )
        if smooth:
            grad = gaussian_filter1d(grad, sigma=2)
        weight = grad + epsilon

        # build normalized CDF
        cdf = np.cumsum(weight)
        cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])

        # invert the cdf to get a refined grid
        inv_cdf = interp1d(cdf, gridNorm)
        gridNorm_adaptive = inv_cdf(gridNorm)

        # mapping back onto real grid
        self.grid_adaptive = (grid.max() - grid.min()) * gridNorm_adaptive + grid.min()

        # enforce minimum grid size
        dx = np.diff(self.grid_adaptive)
        if np.any(dx < minval):
            print(f"[make_adaptive_grid] Warning: grid spacing < dx_min ({minval:.2e}). Switching to semi-uniform.")
            self.grid_adaptive = grid

    def interp_field(self, field):
        """ Interpolate field onto adaptive grid """
        return np.interp(self.grid_adaptive, self.grid, field)


    def plot(self, field):
        """ plotting the new and old fields """
        field_interp = self.interp_field(field)
        fig, ax = plt.subplots()
        ax.plot(self.grid, field, marker="|", label='uniform grid')
        ax.plot(self.grid_adaptive, field_interp, marker="|", label="adaptive grid")
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("grid")
        ax.set_ylabel("field")
        ax.set_title("uniform-vs-adaptive grid")
    

class WaveEqn:
    def __init__(self,
                 grid: np.array,               # grid
                 cs: float,                    # sound speed coefficient
                 bc_lower: str='constant:0',   # lower boundary condition
                 bc_upper: str='constant:0',   # upper boundary condition
                 order: int=0,                 # order of equations, 0=cartesian, 1=cylindrical/polar, 2=spherical
                 use_AMR: bool=False,          # use adaptive mesh refinement
                 ):
        """ 1D wave equation in cartesian coordinates """
        ## setting up the ghost grid
        self.npts           = grid.size
        self.grid           = np.copy( grid )
        self.grid_adaptive  = self.grid

        ## setting up the ghost grid
        self.use_AMR         = use_AMR
        self.dr              = self.grid[1] - self.grid[0]
        self.ghostGrid       = np.zeros(self.npts + 2)
        self.ghostGrid[1:-1] = self.grid
        self.ghostGrid[0]    = grid[0] - self.dr
        self.ghostGrid[-1]   = grid[-1] + self.dr
        self.u_ghost         = np.zeros_like(self.ghostGrid)
        self.dudt_ghost      = np.zeros_like(self.ghostGrid)
        
        ## if cylindrical/polar or spherical coordinates are used, ensure that the ghost grid lower bound is greater than zero
        self.order = order
        if (order == 1 or order == 2) and self.ghostGrid[0] <= 0:
            raise Exception(f"For cylindrical/polar or spherical coordinates, the ghost grid must be > 0. Lower bound={self.ghostGrid[0]}.")
        
        ## generate boundary conditions
        self.cs   = cs
        self.bc_u = GenerateBCs1D(bc_lower, bc_upper)
        

    def __call__(self, t: np.array, y: np.array):
        """ Right hand side of the diffusion equation """
        u    = np.copy(y[:self.npts])
        dudt = np.copy(y[self.npts:])

        ## using adaptive mesh refinement to map to new grid
        if self.use_AMR:
            amr = AdaptiveGrid(self.grid_adaptive, u, epsilon=2.0)
            self.grid_adaptive = amr.grid_adaptive
            u_adaptive = amr.interp_field(u)
            dudt_adaptive = amr.interp_field(dudt)
        else:
            u_adaptive = u
            dudt_adaptive = dudt

        # mapping to the ghost grid
        self.ghostGrid[1:-1] = self.grid_adaptive
        self.u_ghost[1:-1] = u_adaptive
        ## second order spatial difference
        self.ds  = (self.ghostGrid[2:] - self.ghostGrid[:-2]) / 2
        self.ds2 = self.ds**2

        ## apply boundary conditions
        self.bc_u(self.u_ghost, self.ghostGrid)
        
        ## RHS = d^2 (r^alpha * rho) / dr^2 - d (alpha * r^(alpha-1) * rho) / dr
        rAlphaRho = self.u_ghost * self.ghostGrid ** self.order
        alphaRalpham1U = self.order * self.u_ghost * self.ghostGrid ** (self.order - 1)
        term1 = (rAlphaRho[2:] - 2 * rAlphaRho[1:-1] + rAlphaRho[:-2]) / self.ds2
        term2 = (alphaRalpham1U[2:] - alphaRalpham1U[:-2]) / (2 * self.ds)  # use central differencing
        rhs = term1 - term2
        
        #return rhs
        return np.concatenate((dudt_adaptive, rhs))
    

if __name__ == '__main__':
    #%% set up the grid
    tGrid = np.linspace(0, 6, num=300)
    rGrid  = np.linspace(0.1, 10, num=300)
    ## generate initial conditions
    u0    = np.zeros_like(rGrid)
    dudt0 = np.zeros_like(rGrid)
    u0[np.logical_and(rGrid > 4.5, rGrid < 5.5)] = 2
    y0 = np.concatenate((u0, dudt0))

    ## solve the initial value problem
    wave = WaveEqn(rGrid, 4.0,
                   bc_lower='constant:0',
                   bc_upper='constant:0',
                   order=0
                   )
    res = solve_ivp(wave, [tGrid.min(), tGrid.max()], y0, 
                    method='RK45', t_eval=tGrid)
    
    times = res.t 
    u_t   = res.y[:rGrid.size]
    dudt_t = res.y[rGrid.size:]
    
    #%% plotting at discrete times
    outevery = 20
    fig, ax = plt.subplots()
    for it, tTemp in enumerate(times):
        if it % outevery == 0:
            ax.plot(rGrid, u_t[:,it], label=f"time={tTemp:.3f}s", marker="|")
    ax.grid(True)
    ax.legend()

    ## comparing the grids using static mesh vs AMR
    test_it = 100
    amr = AdaptiveGrid(rGrid, u_t[:,test_it], epsilon=2.0)
    amr.plot(u_t[:,test_it])

    #%% density plots
    extent = [tGrid.min(), tGrid.max(), rGrid.min(), rGrid.max()]
    fig, ax = plt.subplots()
    
    cset = ax.pcolormesh(tGrid, rGrid, u_t)
    ax.set_title("Pressure vs distance and time")
    fig.colorbar(cset, ax=ax)
    ax.set_ylabel("distance (m)")
    ax.set_xlabel("Time (s)")