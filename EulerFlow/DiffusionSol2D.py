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
                 grid: tuple,                   # grid
                 diffCoef: float,               # diffusion coefficient
                 bc_lower: str='constant:2',    # lower boundary condition
                 bc_upper: str='constant:1',    # upper boundary condition
                 order: int=0,                  # order of equations, 0=cartesian, 1=cylindrical/polar, 2=spherical
                 ):
        """ 1D diffusion equation in cartesian coordinates """
        ## setting up the ghost grid
        self.nx, self.ny    = grid[0].size, grid[1].size
        self.shape          = (self.nx, self.ny)
        self.X, self.Y      = grid
        self.dX             = self.X[1] - self.X[0]
        self.dY             = self.Y[1] - self.Y[0]
        
        self.ghostX         = np.zeros(self.nx + 2)
        self.ghostX[1:-1]   = self.X
        self.ghostX[0]      = self.X[0] - self.dX
        self.ghostX[-1]     = self.X[-1] + self.dX
        
        self.ghostY         = np.zeros(self.ny + 2)
        self.ghostY[1:-1]   = self.Y
        self.ghostY[0]      = self.Y[0] - self.dY
        self.ghostY[-1]     = self.Y[-1] + self.dY
        self.ghostMeshX, self.ghostMeshY = np.meshgrid(self.ghostX, self.ghostY)
        
        self.u_ghost         = np.zeros((self.nx + 2, self.ny + 2))
        ## if cylindrical/polar or spherical coordinates are used, ensure that the ghost grid lower bound is greater than zero
        self.order = order
        if (order == 1 or order == 2) and self.ghostGrid[0] <= 0:
            raise Exception(f"For cylindrical/polar or spherical coordinates, the ghost grid must be > 0. Lower bound={self.ghostGrid[0]}.")
        
        ## second order spatial difference
        self.dx  = (self.ghostX[2:] - self.ghostX[:-2]) / 2
        self.dx2 = self.dx**2
        self.dy  = (self.ghostY[2:] - self.ghostY[:-2]) / 2
        self.dy2 = self.dy**2
        
        ## generate boundary conditions
        self.diffCoef = diffCoef
        #self.bc_u = GenerateBCs1D(bc_lower, bc_upper)
        
    def createICs(self, u0: np.array):
        """ Create initial conditions based on initial value """
        return u0.reshape(-1)
    
    def conv2Primatives(self, u_res: np.array):
        """ Convert result to primative variables """
        ntimes = u_res.shape[1]
        u_t = u_res.reshape(self.nx, self.ny, ntimes)
        return u_t
        
    def __call__(self, t: np.array, u_arr: np.array):
        """ Right hand side of the diffusion equation """
        u = u_arr.reshape(self.shape)
        self.u_ghost[1:-1,1:-1] = u / self.X ** self.order

        ## apply boundary conditions
        #self.bc_u(self.u_ghost, self.ghostGrid)
        
        ## RHS = d^2 (r^alpha * rho) / dr^2 - d (alpha * r^(alpha-1) * rho) / dr
        #rAlphaRho = self.u_ghost * self.ghostGrid ** self.order
        #alphaRalpham1U = self.order * self.u_ghost * self.ghostGrid ** (self.order - 1)
        #term1 = (rAlphaRho[2:] - 2 * rAlphaRho[1:-1] + rAlphaRho[:-2]) / self.ds2
        #term2 = (alphaRalpham1U[2:] - alphaRalpham1U[:-2]) / (2 * self.ds)  # use central differencing
        #rhs = term1 - term2
        d2Udx2 = (self.u_ghost[2:,1:-1] - 2 * self.u_ghost[1:-1,1:-1] + self.u_ghost[:-2,1:-1]) / self.dx2
        d2Udy2 = (self.u_ghost[1:-1,2:] - 2 * self.u_ghost[1:-1,1:-1] + self.u_ghost[1:-1,:-2]) / self.dy2
        rhs = self.diffCoef * (d2Udx2 + d2Udy2)
        return rhs.reshape(-1)
    

if __name__ == '__main__':
    #%% set up the grid
    ordr  = 0
    tGrid = np.linspace(0, 2, num=50)
    xGrid = np.linspace(0, 10, num=100)
    yGrid = np.linspace(0, 10, num=100)
    X, Y  = np.meshgrid(xGrid, yGrid)
    nx = len(xGrid)
    ny = len(yGrid)

    u0_arr = np.arange(nx*ny)
    u0_2d  = u0_arr.reshape(nx, ny) / u0_arr.max()

    DiffEx = Diffusion((xGrid, yGrid), 1)
    y0     = DiffEx.createICs(u0_2d)

    res = solve_ivp(DiffEx, [tGrid.min(), tGrid.max()], y0=y0,
                    t_eval=tGrid )
    
    u_t = DiffEx.conv2Primatives(res.y)

    fig = plt.figure()
    plt.pcolormesh(X, Y, u_t[:,:,0])
    plt.title(f"Diffusion at t={tGrid.min():.2f}s")
    
    fig = plt.figure()
    plt.pcolormesh(X, Y, u_t[:,:,-1])
    plt.title(f"Diffusion at t={tGrid.max():.2f}s")
    