#!/usr/bin/env python
"""
Author: Hugh Morgan
Date: 2024-08-26
Description: Solve the 1D Euler equations in cartesian, cylindrical, and polar coordinates using the Jameson-Shmidt Turkel numerical scheme in 1D.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from BoundaryConditions import GenerateBCs1D, validBCs
from JamesonShmidtTurkel import JST_DissipFlux, JST_2ndOrderEulerFlux


class EulerSolQuasi1D:
    def __init__(self,
                 grid: np.array,                # computational grid
                 S_x: np.array,                 # area ratio as a function of axial distance
                 alpha: list=[0.5, 0.5],        # dissipative flux terms for spatial differencing
                 beta: list=[0.25, 0.5],        # dissipative flux terms for spatial differencing
                 gamma: float=1.4,              # ratio of specific heats
                 Rspec__JpkgK: float=287.05287, # Ideal gas constant, J/kg-K
                 bcs: dict={'u' : ['gradient:0', 'gradient:0'],
                            'p' : ['reflective', 'gradient:0'],
                            'T' : ['gradient:0', 'gradient:0']
                            }
                 ):
        """
        Solve Euler's system of equations describing the behavior of inviscid, compressible flow by reducing to a system of ordinary differential equations.
        """
        self.grid = grid
        self.S_x  = S_x
        self.R_x  = np.sqrt(S_x)
        self.size = grid.size
        self.gamma = gamma
        ## ideal gas constant
        self.Rspec__JpkgK = Rspec__JpkgK

        ## spatial differencing and numerical diffusion coefficients
        self.alpha = alpha
        self.beta  = beta

        ## setting up ghost grid
        self.dr              = grid[1] - grid[0]
        self.ghostGrid       = np.zeros(self.grid.size + 2)
        self.ghostGrid[1:-1] = self.grid
        self.ghostGrid[0]    = grid[0] - self.dr
        self.ghostGrid[-1]   = grid[-1] + self.dr

        self.ghostRho = np.ones_like(self.ghostGrid)
        self.ghostU   = np.zeros_like(self.ghostGrid)
        self.ghostE   = np.zeros_like(self.ghostGrid)
        self.ghostP   = np.ones_like(self.ghostGrid)
        self.ghostT   = np.zeros_like(self.ghostGrid)

        ## cross-sectional area and it's dervivateive as a function of axial distance, ghost grid
        self.ghostS_x   = np.interp(self.ghostGrid, self.grid, self.S_x)
        ## use central differencing to determing the derivative
        self.dSdx_x     = (self.S_x[2:] - self.S_x[:-2]) / (self.grid[2:] - self.grid[:-2])
        self.ghostdSdx  = np.interp(self.ghostGrid, self.grid[1:-1], self.dSdx_x)

        ## defining boundary conditions
        self.__uBC = GenerateBCs1D(bcs['u'][0], bcs['u'][1])
        self.__pBC = GenerateBCs1D(bcs['p'][0], bcs['p'][1])
        self.__TBC = GenerateBCs1D(bcs['T'][0], bcs['T'][1])


    def createICs(self, 
                  rho0: np.array, 
                  v0: np.array, 
                  p0: np.array):
        """ Convert primative variables into initial conditions - W0 """
        ## using equations of state, calculate internal energy
        E0 = p0 / (rho0 * (self.gamma - 1)) + 0.5 * v0**2
        W0 = [rho0 * self.S_x,
              rho0 * v0 * self.S_x,
              rho0 * E0 * self.S_x]
        return np.concatenate(W0)
    

    def conv2Primatives(self, W: np.array):
        """ Convert W result to primative values """
        rho   = W[0:self.size,:].T / self.S_x             # first block contains rho
        rho_U = W[self.size:2*self.size,:].T / self.S_x   # second block contains rho*U
        rho_E = W[2*self.size:,:].T / self.S_x            # third block contains rho*E

        rho = rho.T
        U = rho_U.T / rho
        E = rho_E.T / rho
        p = rho * (self.gamma - 1) * (E - 0.5 * U**2)
        return rho, U, E, p
    
    def __call__(self, t, x):
        """
        For the flow in a 1D channel with a given slowly-varying cross-sectional area S(x), we can write the Euler equations as
            d[W*S(x)] / dt + d[F*S(x)] / dx = Q
        where
                [  rho  ]       [   rho*u   ]       [    0    ]
            W = [ rho*u ]   F = [ rho*u*u+p ]   Q = [ p*S'(x) ]
                [ rho*E ]       [  rho*u*H  ]       [    0    ]
        p, rho, u, E and H denote the pressure, density, velocity , total energy and total enthalpy. For a
perfect gas
            E = p / (rho*(gamma-1)) + 1/2 * u^2
            H = E + p / rho
        where gamma is the ratio of specific heats.
        """
        rho   = x[0:self.size] / self.S_x             # first block contains rho
        rho_U = x[self.size:2*self.size] / self.S_x   # second block contains rho*U
        rho_E = x[2*self.size:] / self.S_x            # third block contains rho*E

        ## convert to primatives - Apply equations of staate
        u = rho_U / rho
        E = rho_E / rho
        p = rho * (self.gamma - 1) * (E - u**2)
        #H = E + p / rho
        T = p / (rho * self.Rspec__JpkgK)

        ## ALL FOLLOWING SCALAR VALUES LIE ALONG THE GHOST GRID
        ## apply boundary conditions to u, p and t
        self.ghostU[1:-1] = u
        self.ghostP[1:-1] = p
        self.ghostT[1:-1] = T
        self.__uBC(self.ghostU, self.ghostGrid)
        self.__pBC(self.ghostP, self.ghostGrid)
        self.__TBC(self.ghostT, self.ghostGrid)

        ## apply equations of state to calculate rho, U, and E on the ghost grid
        self.ghostRho = self.ghostP / (self.ghostT * self.Rspec__JpkgK)
        self.ghostE   = self.ghostP / (self.ghostRho * (self.gamma - 1)) + 0.5 * self.ghostU**2
        H = self.ghostE + self.ghostP / self.ghostRho
        cs = np.sqrt( self.gamma * self.ghostP / self.ghostRho)

        ## develop W - state vector variable
        W = [self.ghostRho * self.ghostS_x,
             self.ghostRho * self.ghostU * self.ghostS_x,
             self.ghostRho * self.ghostE * self.ghostS_x
             ]
        
        ## develop F - flux vector variable
        F = [self.ghostRho * self.ghostU * self.ghostS_x,
             (self.ghostRho * self.ghostU**2 + self.ghostP) * self.ghostS_x,
             self.ghostRho * self.ghostU * H * self.ghostS_x
             ]

        ## develop S - source term variable
        S = [np.zeros_like(self.ghostP), 
             self.ghostP * self.ghostdSdx,
             np.zeros_like(self.ghostP) ]
        
        ## calculate second-order Euler flux and dissipation flux
        Qj = JST_2ndOrderEulerFlux(F)
        Dj = JST_DissipFlux(W, self.ghostP, self.ghostU, cs, self.alpha, self.beta)
        
        Rj = []
        for stemp, qtemp, dtemp in zip(S, Qj, Dj):
            Rj.append( stemp[1:-1] -1 / self.dr * (qtemp - dtemp) )
        
        return np.concatenate(Rj)


class Rocket1D:
    def __init__(self,
                 grid__m: np.array,         # 1D axial grid, meters
                 Sx__m2: np.array,          # area as a function of axial distance, m^2
                 rho0__kgpm3: np.array,     # ambient air density, kg/m^3
                 P0__Pa: np.array,          # initial pressure
                 v0__mps: np.array,         # ambient air density, m/s
                 times__s: float,           # solution times, seconds
                 ScaleLen__m: float=1,      # length scale, meters
                 ScaleTemp__K: float=300,   # scale temperature, Kelvin
                 gamma: float=1.4,          # ratio of specific heats, N/A
                 Rspec__JpkgK: float=287.05287, # Ideal gas constant, J/kg-K
                 ):
        """
        Convert the parameters to nondimensional form, for speed and numerical stability.
        """
        self.gamma      = gamma
        self.nGridPts   = grid__m.size

        ## dimensionless parameters: scale using rhoScale, PScale, and ScaleLen__m
        self.ScaleLen__m        = ScaleLen__m
        self.PScale__Pa         = P0__Pa.min()
        self.rhoScale__kgpm3    = rho0__kgpm3.min()
        self.UScale             = np.sqrt(self.PScale__Pa / self.rhoScale__kgpm3)

        self.grid   = grid__m / ScaleLen__m
        self.times  = times__s / ScaleLen__m * self.UScale
        self.SxStar = Sx__m2 / ScaleLen__m**2
        ## NEED TO SCALE IDEAL GAS CONSTANT
        self.RspecStar = Rspec__JpkgK / self.UScale**2 * ScaleTemp__K

        ## setting the initial conditions
        self.rho0 = rho0__kgpm3 / self.rhoScale__kgpm3
        self.p0   = P0__Pa / self.PScale__Pa
        self.v0   = v0__mps / self.UScale

        ## time and grid in dimensional/metric scale
        self.r__m = self.grid * ScaleLen__m
        self.t__s = self.times * ScaleLen__m / self.UScale
        self.T__s, self.R__m = np.meshgrid(self.t__s, self.r__m)
    
    def solve(self,
              BoundaryConditions: dict,
              method: str='RK45'):
        """ Solve the system of partial differential equations using scipy.integrate.solve_ivp"""
        self.ODEs = EulerSolQuasi1D(self.grid, self.SxStar, gamma=self.gamma,
                                    bcs=BoundaryConditions, Rspec__JpkgK=self.RspecStar)
        self.y0 = self.ODEs.createICs(self.rho0, self.v0, self.p0)
        t_range = [self.times.min(), self.times.max()]
        r_range = [self.grid.min(), self.grid.max()]
        
        print(f"Solving the Euler Equation as a system of ODES. \nt_range={t_range}(dimensionless) \nnGridPts={self.nGridPts}\nr_range={r_range}(dimensionless)")
        res = solve_ivp(self.ODEs, t_range, self.y0,
                        t_eval=self.times,
                        method=method)
        
        rhoStar_t, uStar_t, eStar_t, pStar_t = self.ODEs.conv2Primatives(res.y)
        self.rho = rhoStar_t * self.rhoScale__kgpm3
        self.u   = uStar_t * np.sqrt(self.PScale__Pa / self.rhoScale__kgpm3)
        self.p   = pStar_t * self.PScale__Pa
        self.E   = self.p / (self.rho * (self.gamma - 1)) + 0.5 * self.u**2
        return res

    def plotAreaRelation(self):
        """ Plot the radius, area, and area derivative with respect to axial distance """
        fig, ax = plt.subplots()
        color = 'tab:blue'
        p0 = ax.plot(self.ODEs.grid, self.ODEs.R_x, label='Area', color=color)
        ax.set_ylim(0, self.ODEs.R_x.max())
        ax.grid(True)
        ax.set_xlabel('distance (m)')
        ax.set_ylabel('radius(m)', color=color)
        ax.tick_params(axis='y', labelcolor=color)

        color = 'tab:red'
        ax2 = ax.twinx()
        p2 = ax2.plot(self.ODEs.ghostGrid, self.ODEs.ghostS_x, label='dArea/dx', color=color)
        ax2.set_ylabel('dArea/dx (m)', color=color)
        ax2.tick_params(axis='y', labelcolor=color)


    def dispFields(self):
        """ Display the field variables as functions of space and time """
        fig, axes = plt.subplots(nrows=2, ncols=2)
        def field(ax, val, desc, logPlot=False):
            if logPlot: norm='log'
            else: norm='linear'
            cs = ax.pcolormesh(1000*self.T__s, self.R__m, val,
                           norm=norm, cmap='jet')
            fig.colorbar(cs, ax=ax, label=desc)

        field(axes[0][0], self.rho, r"density ($kg/m^3$)")
        field(axes[1][0], self.u,   r"velocity ($m/s$)")
        field(axes[0][1], self.p,   r"Pressure ($Pa$)", logPlot=True)
        field(axes[1][1], self.E,   r"Total Energy ($J$)", logPlot=True)
        axes[1][0].set_xlabel('time (ms)')
        axes[1][1].set_xlabel('time (ms)')
        axes[0][0].set_ylabel('distance (m)')
        axes[1][0].set_ylabel('distance (m)')


    def plotDiscTimes(self, n_plots=20):
        """ plots at discrete times """
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,)
        def linPlot(ax, var, desc, log=False):
            ## plot the solution at discrete times
            for it, tTemp in enumerate(self.times):
                if (it + n_plots) % n_plots == 0:
                    t__ms = 1000 * tTemp
                    if log:
                        ax.semilogy(self.r__m, var[:,it], label=f"t={t__ms:.2f}ms")
                    else:
                        ax.plot(self.r__m, var[:,it], label=f"t={t__ms:.2f}ms")
            ax.set_ylabel(desc)
            ax.grid(True)

        linPlot(axes[0][0], self.rho, 'density (kg/m^3)')
        linPlot(axes[1][0], self.u,   'velocity (m/s)')
        linPlot(axes[0][1], self.p,   'pressure (Pa)', log=True)
        linPlot(axes[1][1], self.E,   'total energy (J)', log=True)
        axes[1][0].set_xlabel('r (m)')
        axes[1][1].set_xlabel('r (m)')
        axes[1][0].legend()


if __name__ == '__main__':
    ## final time
    times = np.linspace(0, 0.002, num=100)
    ## setting up the grid
    x = np.linspace(0,10, num=300)
    ## area vs radial distance
    Area_x = np.ones_like(x)
    Area_x[x < 5]  = 1 + 1.5 * (1 - x[x<5]/5)**2
    Area_x[x >= 5] = 1 + 0.5 * (1 - x[x>=5]/5)**2

    ## setting initial conditions
    P0   = 101325 * np.ones_like(x)
    u0   = 83 * np.ones_like(x)
    rho0 = 1.225 * np.ones_like(x)
    P0[x < 2] = 4 * 101325
    ## boundary conditions
    BCs = {'u' : ['gradient:0', 'extrapolated'], #['constant:0.28859435', 'extrapolated'],
           'p' : ['gradient:0', 'gradient:0'], #['constant:10', 'constant:4'],
           'T' : ['gradient:0', 'extrapolated'] #['constant:0.96505', 'extrapolated']
           }
    
    #%% setting up the Rocket problem
    Engine = Rocket1D(x, Area_x, rho0, P0, u0, times)

    res = Engine.solve(BCs)
    Engine.plotAreaRelation()
    Engine.dispFields()
    Engine.plotDiscTimes()