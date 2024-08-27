#!/usr/bin/env python
"""
Author: Hugh Morgan
Date: 2024-08-26
Description: Solve the 1D Euler equations in cartesian, cylindrical, and polar coordinates using the Jameson-Shmidt Turkel numerical scheme in 1D.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def d3dx3_fwd(W: np.array):
    ## third-order spatial differencing, FWD
    d3Wdx3 = W[1:] - W[:-1]
    return d3Wdx3

def d3dx3_bkwd(W: np.array):
    ## third-order spatial differencing, BCKWRDS
    d3Wdx3 = W[1:] - W[:-1]


def JST_2ndOrderEulerFlux(F: np.array):
    """ Calculate the second-order Euler flux """
    Q_j = []
    for ftemp in F:
        hbar_jphalf = (ftemp[2:] + ftemp[1:-1]) / 2
        hbar_jmhalf = (ftemp[1:-1] + ftemp[:-2]) / 2
        Q_j.append( hbar_jphalf - hbar_jmhalf)
    return Q_j

def JST_DissipFlux(W: np.array,
                   F: np.array,
                   alpha: list,
                   beta: list,
                   ):
    """ Calculate dissipation flux using JST method"""
    D_j = []
    for wtemp, ftemp in zip(W, F):

        D_j.append(wtemp + ftemp)
    return D_j

class EulerSol:
    def __init__(self,
                 grid: np.array,            # computational grid
                 order: int=0,              # order of equations, 0=cartesian, 1=cylindrical/polar, 2=spherical
                 alpha: list=[0.5, 0.5],    # dissipative flux terms for spatial differencing
                 beta: list=[0.25, 0.5],    # dissipative flux terms for spatial differencing
                 gamma: float=1.4,          # ratio of specific heats
                 ):
        """
        Solve Euler's system of equations describing the behavior of inviscid, compressible flow by reducing to a system of ordinary differential equations.
        """
        self.grid = grid
        self.size = grid.size
        self.order = order
        self.gamma = gamma

        ## spatial differencing and numerical diffusion coefficients
        self.alpha = alpha
        self.beta  = beta

        ## setting up ghost grid
        self.dr = grid[1] - grid[0]
        self.ghostGrid       = np.zeros(self.grid.size + 2)
        self.ghostGrid[1:-1] = self.grid
        self.ghostGrid[0]    = grid[0] - self.dr
        self.ghostGrid[-1]   = grid[-1] + self.dr

        self.ghostRho = np.ones_like(self.ghostGrid)
        self.ghostU   = np.zeros_like(self.ghostGrid)
        self.ghostE   = np.zeros_like(self.ghostGrid)

    def createICs(self, 
                  rho0: np.array, 
                  v0: np.array, 
                  p0: np.array):
        """ Convert primative variables into initial conditions - W0 """
        ## using equations of state, calculate internal energy
        E0 = p0 / (rho0 * (self.gamma - 1)) + 0.5 * v0**2
        W0 = [rho0 * self.grid**self.order,
              rho0 * v0 * self.grid**self.order,
              rho0 * E0 * self.grid**self.order]
        return W0
    

    def conv2Primatives(self, W: np.array):
        """ Convert W result to primative values """
        rho   = W[0:self.size] / self.grid**self.order             # first block contains rho
        rho_U = W[self.size:2*self.size] / self.grid**self.order   # second block contains rho*U
        rho_E = W[2*self.size:] / self.grid**self.order            # third block contains rho*E

        U = rho_U / rho
        E = rho_E / rho
        p = rho * (self.gamma - 1) * (E - 0.5 * U**2)
        return rho, U, E, p
    
    def __call__(self, t, x):
        """
        """
        rho   = x[0:self.size] / self.grid**self.order             # first block contains rho
        rho_U = x[self.size:2*self.size] / self.grid**self.order   # second block contains rho*U
        rho_E = x[2*self.size:] / self.grid**self.order            # third block contains rho*E

        ## convert to primatives
        u = rho_U / rho
        E = rho_E / rho

        ## apply boundary conditions
        self.ghostRho[1:-1] = rho
        self.ghostU[1:-1]   = u 
        self.ghostE[1:-1]   = E
        ## reflective at origin
        self.ghostU[0]   = self.ghostU[1]
        self.ghostE[0]   = self.ghostE[1]
        self.ghostRho[0] = self.ghostRho[1]
        ## transmissive at end
        self.ghostRho[-1] = self.ghostRho[-2]
        self.ghostU[-1]   = self.ghostU[-2]
        self.ghostE[-1]   = self.ghostE[-2]

        ## apply equations of state
        p = self.ghostRho * (self.gamma - 1) * (self.ghostE - 0.5 * self.ghostU**2)
        H = self.ghostE + p / self.ghostRho

        ## develop W - state vector variable
        W = [self.ghostRho * self.ghostGrid**self.order,
             self.ghostRho * self.ghostU * self.ghostGrid**self.order,
             self.ghostRho * self.ghostE * self.ghostGrid**self.order
             ]
        
        ## develop F - flux vector variable
        F = [self.ghostRho * self.ghostU * self.ghostGrid**self.order,
             (self.ghostRho * self.ghostU**2 + p) * self.ghostGrid**self.order,
             self.ghostRho * self.ghostU * H * self.ghostGrid**self.order
             ]

        ## develop S - source term variable
        if self.order == 0:
            S = [0, 0, 0]
        else:
            S = [0, self.order * p * self.ghostGrid**(self.order-1), 0 ]
        
        ## calculate second-order Euler flux and dissipation flux
        Qj = JST_2ndOrderEulerFlux(F)
        Dj = JST_DissipFlux(W, F, self.alpha, self.beta)
        
        for stemp, qtemp, dtemp in zip(S, Qj, Dj):
            Rj = stemp[1:-1] - 1 / self.dr * (qtemp - dtemp)
        
        return np.concatenate(Rj)


if __name__ == '__main__':
    ## define grid
    rMin__m = 0.1
    rMax__m = 10
    nGridPts = 200
    rGrid = np.linspace(rMin__m, rMax__m, num=nGridPts)
    tGrid = np.linspace(0, 0.1, num=nGridPts)

    ## define initial conditions
    P0__Pa = 101325
    PExpl__Pa = 20 * P0__Pa
    rExpl__m  = 4
    rho0__kgpm3 = 1.225
    
    rho0 = rho0__kgpm3 * np.ones_like(rGrid)
    p0   = P0__Pa * np.ones_like(rGrid)
    p0[rGrid < rExpl__m] = PExpl__Pa
    vr0 = np.zeros_like(rGrid)
    order = 0

    ## set up system of equations
    ES = EulerSol(rGrid, order=order)
    y0 = ES.createICs(rho0, vr0, p0)

    res = solve_ivp(ES, y0, method='RK45',
                    t_eval=tGrid, )
    
    rho_t, U_t, E_t, p_t = ES.conv2Primatives(res.y)