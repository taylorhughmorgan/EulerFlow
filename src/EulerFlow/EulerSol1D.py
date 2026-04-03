#!/usr/bin/env python
"""
Author: Hugh Morgan
Date: 2024-08-26
Description: Solve the 1D Euler equations in cartesian, cylindrical, and polar coordinates using the Jameson-Shmidt Turkel numerical scheme in 1D.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from EulerFlow.BoundaryConditions import GenerateBCs1D
from EulerFlow.JamesonShmidtTurkel import JST_DissipFlux, JST_2ndOrderEulerFlux
from math import gamma as gamma_func

class shockReachEnd:
    """ Abort the simulation if the shock reaches the end of the domain """
    def __init__(self, 
                 grid : np.array, 
                 order : int, 
                 gamma : float,
                 terminal : bool):
        self.size = grid.size
        self.grid = grid
        self.order = order
        self.gamma = gamma
        self.terminal = terminal
        self.direction = -1

    def __call__(self, t, x):
        """
        Determine if the domain end is reached by looking at rho, U, and E. 
        If the pressure is > 1/10th P_max then abort the simulation.
        """
        rho   = x[0:self.size] / self.grid**self.order             # first block contains rho
        rho_U = x[self.size:2*self.size] / self.grid**self.order   # second block contains rho*U
        rho_E = x[2*self.size:] / self.grid**self.order            # third block contains rho*E

        ## convert to primatives
        u = rho_U / rho
        E = rho_E / rho
        p = rho * (self.gamma - 1) * (E - 0.5 * u**2)
        if p[-1] / p.max() > 0.5:
            return -1.0
        else:
            return 1.0
    

class EulerSol:
    def __init__(self,
                 grid: np.array,                # computational grid
                 order: int=0,                  # order of equations, 0=cartesian, 1=cylindrical/polar, 2=spherical
                 alpha: list=[0.5, 0.5],        # dissipative flux terms for spatial differencing
                 beta: list=[0.25, 0.5],        # dissipative flux terms for spatial differencing
                 gamma: float=1.4,              # ratio of specific heats
                 rho_limit: float=1e-6,         # limit density from dropping too low
                 bcs: dict={'rho' : ['gradient:0', 'gradient:0'],
                            'u'   : ['gradient:0', 'gradient:0'],
                            'E'   : ['gradient:0', 'gradient:0']
                            }
                 ):
        """
        Solve Euler's system of equations describing the behavior of inviscid, compressible flow by reducing to a system of ordinary differential equations.
        """
        self.grid = grid
        self.size = grid.size
        self.order = order
        self.gamma = gamma
        self.rho_limit = rho_limit
        self.press_limit = np.inf

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

        ## defining boundary conditions
        self.__rhoBC = GenerateBCs1D(bcs['rho'][0], bcs['rho'][1])
        self.__uBC   = GenerateBCs1D(bcs['u'][0], bcs['u'][1])
        self.__eBC   = GenerateBCs1D(bcs['E'][0], bcs['E'][1])


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
        self.press_limit = p0.max()
        return np.concatenate(W0)
    

    def conv2Primatives(self, W: np.ndarray):
        """ Convert W results matrix to primative values """
        rho   = W[0:self.size,:].T / self.grid**self.order             # first block contains rho
        rho_U = W[self.size:2*self.size,:].T / self.grid**self.order   # second block contains rho*U
        rho_E = W[2*self.size:,:].T / self.grid**self.order            # third block contains rho*E

        rho = rho.T
        U = rho_U.T / rho
        E = rho_E.T / rho
        p = rho * (self.gamma - 1) * (E - 0.5 * U**2)
        return rho, U, E, p
    

    def conv_state_to_prim(self, W: np.ndarray):
        """ convert a single state vector to primatives: rho, U, E, and P """
        rho   = W[0:self.size] / self.grid**self.order             # first block contains rho
        rho_U = W[self.size:2*self.size] / self.grid**self.order   # second block contains rho*U
        rho_E = W[2*self.size:] / self.grid**self.order            # third block contains rho*E

        U = rho_U / rho
        E = rho_E / rho
        p = rho * (self.gamma - 1) * (E - 0.5 * U**2)
        # apply limits
        p[p > self.press_limit] = self.press_limit
        rho[rho < self.rho_limit] = self.rho_limit
        return rho, U, E, p


    def calc_dt(self, W: np.ndarray, CFL: float=0.3):
        """ Calculate the maximum time step based on the grid size and max wave speed.
            dt = CFL * dx / max_wave_speed = CFL * dx / (|u| + c_s)
        """
        dx_min = self.dr.min()

        # convert y to primitives
        rho, U, E, p = self.conv_state_to_prim(W)

        # calculate sound speed
        cs = np.sqrt(self.gamma * p / rho)

        # find the maximum wave speed across the domain
        wave_speed = np.abs(U) + cs
        return CFL * dx_min / wave_speed.max()

    def __call__(self, t, x):
        """
        """
        ## convert to primatives
        rho, u, E, p = self.conv_state_to_prim(x)

        ## apply boundary conditions
        self.ghostRho[1:-1] = rho
        self.ghostU[1:-1]   = u 
        self.ghostE[1:-1]   = E
        self.__rhoBC(self.ghostRho, self.ghostGrid)
        self.__uBC(self.ghostU, self.ghostGrid)
        self.__eBC(self.ghostE, self.ghostGrid)

        ## apply equations of state
        p = self.ghostRho * (self.gamma - 1) * (self.ghostE - 0.5 * self.ghostU**2)
        H = self.ghostE + p / self.ghostRho
        cs = np.sqrt( self.gamma * p / self.ghostRho)

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
            S = [0, 
                 self.order * p[1:-1] * self.ghostGrid[1:-1]**(self.order-1), 
                 0 ]
        
        ## calculate second-order Euler flux and dissipation flux
        Qj = JST_2ndOrderEulerFlux(F)
        Dj = JST_DissipFlux(W, p, self.ghostU, cs, self.alpha, self.beta)
        
        Rj = []
        for stemp, qtemp, dtemp in zip(S, Qj, Dj):
            Rj.append( stemp -1 / self.dr * (qtemp - dtemp) )
        
        return np.concatenate(Rj)



if __name__ == '__main__':
    from tqdm import tqdm
    from EulerFlow.mathutils import hyperbolic_step
    #%% solving the euler equations for a discrete shock
    orders = 2
    ## define grid
    DomainLen__m = 10   # size of the domain
    rMin__m = 0.1
    tMax__s = 1.25
    nGridPts = 500
    rGrid = np.linspace(rMin__m, DomainLen__m, num=nGridPts)
    tGrid = np.linspace(0, tMax__s, num=nGridPts)

    ## define initial conditions
    P0__Pa      = 1
    PExpl__Pa   = 100 * P0__Pa
    rExpl__m    = 1.5
    rho0__kgpm3 = 1
    
    rho0 = rho0__kgpm3 * np.ones_like(rGrid)
    ## setting the initial conditions using hyperbolic tangent instead of step for stability
    p0   = PExpl__Pa * hyperbolic_step(rGrid, rExpl__m, delta=0.1)
    ## setting ambient pressure
    p0[p0 < P0__Pa] = P0__Pa
    vr0  = np.zeros_like(rGrid)

    ## set up system of equations
    ES = EulerSol(rGrid, order=orders)
    # create initial conditions
    y0 = ES.createICs(rho0, vr0, p0)

    # array to save whole solution
    rho_all = np.zeros((rGrid.size, tGrid.size))
    U_all = np.zeros_like(rho_all)
    E_all = np.zeros_like(rho_all)
    p_all = np.zeros_like(rho_all)

    # loop through integration and time time steps based on CFL
    t = 0

    with tqdm(total=tMax__s, desc="Solving", unit="t", bar_format="{l_bar}{bar}| {n:.3f}/{total:.2f} [{elapsed}<{remaining}]") as pbar:
        while t < tMax__s:
            # calculate max time step
            dt_max = ES.calc_dt(y0)
            t_next = min(t + dt_max, tMax__s)

            # isolating the solution using a mask
            mask = (tGrid > t) & (tGrid < t_next)
            t_sample = tGrid[mask]

            # integrate the system of equations
            res = solve_ivp(ES, 
                            [t, t_next], 
                            y0, 
                            max_step=dt_max,
                            #first_step=dt_max,
                            method='RK45',
                            dense_output=True
                            )
            
            if len(t_sample) > 0:
                # extracting primatives
                rho_t, U_t, E_t, p_t = ES.conv2Primatives(res.sol(t_sample))
                
                # adding to big array
                rho_all[:,mask] = rho_t
                U_all[:,mask] = U_t
                E_all[:,mask] = E_t
                p_all[:,mask] = p_t
                
            # save final time and initial conditions for next iteration
            t = res.t[-1]
            y0 = res.y[:,-1]

            # updating the progress bar
            dt_actual = min(dt_max, tMax__s - t)
            pbar.update(dt_actual)
            pbar.set_postfix({"t": f"{t:.2f}", "dt": f"{dt_actual:.2e}"})

    ## density plots
    extent = [tGrid.min(), tGrid.max(), rGrid.min(), rGrid.max()]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    def densPlot(ax, var, desc, log=False):
        ## create a density plot of a given variable
        if log:
            cset = ax.pcolormesh(tGrid, rGrid, var, norm='log', cmap='jet')
        else:
            cset = ax.pcolormesh(tGrid, rGrid, var, norm='linear', cmap='jet')
        ax.set_title(desc)
        fig.colorbar(cset, ax=ax)
    
    densPlot(axes[0][0], rho_all, 'density (kg/m^3)')
    densPlot(axes[0][1], U_all,   'radial velocity (m/s)')
    densPlot(axes[1][0], E_all,   'total energy (J)')
    densPlot(axes[1][1], p_all,   'pressure (Pa)')
    axes[1][0].set_xlabel('time (ms)')
    axes[1][1].set_xlabel('time (ms)')
    axes[0][0].set_ylabel('r (m)')
    axes[1][0].set_ylabel('r (m)')

    ## plots at discrete times
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,)
    def linPlot(ax, var, desc, n_plots=20, log=False):
        ## plot the solution at discrete times
        for it, tTemp in enumerate(tGrid):
            if (it + n_plots) % n_plots == 0:
                t__ms = 1000 * tTemp
                if log:
                    ax.semilogy(rGrid, var[:,it], label=f"t={t__ms:.2f}ms")
                else:
                    ax.plot(rGrid, var[:,it], label=f"t={t__ms:.2f}ms")
        ax.set_ylabel(desc)
        ax.grid(True)

    linPlot(axes[0][0], rho_all, 'density (kg/m^3)')
    linPlot(axes[0][1], U_all,   'radial velocity (m/s)')
    linPlot(axes[1][0], E_all,   'total energy (J)', log=True)
    linPlot(axes[1][1], p_all,   'pressure (Pa)', log=True)
    axes[1][0].set_xlabel('r (m)')
    axes[1][1].set_xlabel('r (m)')
    axes[1][0].legend()
    #%%