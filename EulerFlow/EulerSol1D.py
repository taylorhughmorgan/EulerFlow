#!/usr/bin/env python
"""
Author: Hugh Morgan
Date: 2024-08-26
Description: Solve the 1D Euler equations in cartesian, cylindrical, and polar coordinates using the Jameson-Shmidt Turkel numerical scheme in 1D.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def d3dx3_fwd(y: np.array):
    ## third-order spatial differencing, forwards
    d3ydx3 = np.zeros(y.size - 2)
    d3ydx3[:-1] = (y[3:] - 3*y[2:-1] + 3*y[1:-2] - y[:-3])
    d3ydx3[-1]  = (y[-1] - 3*y[-2] + 3*y[-3] - y[-4]) #backward differencing on last cell
    return d3ydx3

def d3dx3_bkwd(y: np.array):
    ## third-order spatial differencing, backwards
    d3ydx3 = np.zeros(y.size - 2)
    d3ydx3[0:1] = (y[3:4] - 3*y[2:3] + 3*y[1:2] - y[0:1]) # forward differencing on first cells
    d3ydx3[2:]  = (y[3:-1] - 3*y[2:-2] + 3*y[1:-3] - y[:-4]) 
    return d3ydx3


def JST_2ndOrderEulerFlux(F: np.array):
    """ Calculate the second-order Euler flux """
    Q_j = []
    for ftemp in F:
        hbar_jphalf = (ftemp[2:] + ftemp[1:-1]) / 2
        hbar_jmhalf = (ftemp[1:-1] + ftemp[:-2]) / 2
        Q_j.append( hbar_jphalf - hbar_jmhalf)
    return Q_j

def JST_DissipFlux(W: np.array,
                   p: np.array,
                   u: np.array,
                   cs: np.array,
                   alpha: list,
                   beta: list,
                   ):
    """ Calculate dissipation flux using JST method"""
    D_j = []
    ## pressure dissipation term: central differencing
    nu_j = np.zeros(p.size - 1)
    nu_j[:-1] = np.abs( (p[2:] - 2*p[1:-1] + p[:-2]) / (p[2:] + 2*p[1:-1] + p[:-2]) )
    nu_j[-1]  = np.abs( (p[-1] - 2*p[-2] + p[-3]) / (p[-1] + 2*p[-2] + p[-3]) )

    ## maximum wave speed of the system
    R_jphalf = ( (np.abs(u) + cs)[2:] + (np.abs(u) + cs)[1:-1] ) / 2
    R_jmhalf = ( (np.abs(u) + cs)[1:-1] + (np.abs(u) + cs)[:-2] ) / 2

    for wtemp in W:
        ## first derivative of the state vector
        deltaW_jphalf = wtemp[2:] - wtemp[1:-1]
        deltaW_jmhalf = wtemp[1:-1] - wtemp[:-2]
        ## third derivative of the state vector, fwd differencing for j+1/2 and backward for j-1/2
        delta3W_jphalf = d3dx3_fwd(wtemp)
        delta3W_jmhalf = d3dx3_bkwd(wtemp)

        ## dissipative coefficients, S_(j-1/2) MIGHT NEED FIXING
        S_jphalf = np.max( np.stack([nu_j[1:], nu_j[:-1]]), axis=0)
        S_jmhalf = np.max( np.stack([nu_j[1:], nu_j[:-1]]), axis=0)

        eps2_jphalf = np.min( np.stack([alpha[0] * np.ones_like(S_jphalf), alpha[1] * S_jphalf]), axis=0)
        eps2_jmhalf = np.min( np.stack([alpha[0] * np.ones_like(S_jmhalf), alpha[1] * S_jmhalf]), axis=0)
        eps4_jphalf = np.max( np.stack([np.zeros_like(eps2_jphalf), beta[0] - beta[1] * eps2_jphalf]), axis=0)
        eps4_jmhalf = np.max( np.stack([np.zeros_like(eps2_jmhalf), beta[0] - beta[1] * eps2_jmhalf]), axis=0)
        
        ## artificial dissipation terms
        d_jphalf = eps2_jphalf * R_jphalf * deltaW_jphalf - eps4_jphalf * R_jphalf * delta3W_jphalf
        d_jmhalf = eps2_jmhalf * R_jmhalf * deltaW_jmhalf - eps4_jmhalf * R_jmhalf * delta3W_jmhalf
        D_j.append(d_jphalf - d_jmhalf)

    return D_j

class EulerSol:
    def __init__(self,
                 grid: np.array,                # computational grid
                 order: int=0,                  # order of equations, 0=cartesian, 1=cylindrical/polar, 2=spherical
                 alpha: list=[0.5, 0.5],        # dissipative flux terms for spatial differencing
                 beta: list=[0.25, 0.5],        # dissipative flux terms for spatial differencing
                 gamma: float=1.4,              # ratio of specific heats
                 bc_lower: str='reflective',    # boundary condition on the lower bound
                 bc_upper: str='transmissive'   # boundary condition on the upper bound
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

        ## defining boundary conditions
        credible_bcs = ['reflective', 'transmissive', 'extrapolated']
        if bc_upper not in credible_bcs:
            raise Exception(f"{bc_upper} not a credible Boundary Condition. Use {credible_bcs}.")
        if bc_lower not in credible_bcs:
            raise Exception(f"{bc_lower} not a credible Boundary Condition. Use {credible_bcs}.")
        self.bc_lower = bc_lower
        self.bc_upper = bc_upper


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
        return np.concatenate(W0)
    

    def conv2Primatives(self, W: np.array):
        """ Convert W result to primative values """
        rho   = W[0:self.size,:].T / self.grid**self.order             # first block contains rho
        rho_U = W[self.size:2*self.size,:].T / self.grid**self.order   # second block contains rho*U
        rho_E = W[2*self.size:,:].T / self.grid**self.order            # third block contains rho*E

        rho = rho.T
        U = rho_U.T / rho
        E = rho_E.T / rho
        p = rho * (self.gamma - 1) * (E - 0.5 * U**2)
        return rho, U, E, p
    

    def __applyBCs(self):
        """ Apply Boundary conditions. """
        ## boundary conditions on the lower bound
        if self.bc_lower == 'reflective':
            self.ghostU[0]   = -self.ghostU[1]
            self.ghostE[0]   = self.ghostE[1]
            self.ghostRho[0] = self.ghostRho[1]
        elif self.bc_lower == 'transmissive':
            self.ghostU[0]   = self.ghostU[1]
            self.ghostE[0]   = self.ghostE[1]
            self.ghostRho[0] = self.ghostRho[1]
        elif self.bc_lower == 'extrapolated':
            self.ghostU[0]   = 2 * self.ghostU[1] - self.ghostU[2]
            self.ghostE[0]   = 2 * self.ghostE[1] - self.ghostE[2]
            self.ghostRho[0] = 2 * self.ghostRho[1] - self.ghostRho[2]
        
        ## boundary conditions on the upper bound
        if self.bc_upper == 'reflective':
            self.ghostU[-1]   = -self.ghostU[-2]
            self.ghostE[-1]   = self.ghostE[-2]
            self.ghostRho[-1] = self.ghostRho[-2]
        elif self.bc_upper == 'transmissive':
            self.ghostU[-1]   = self.ghostU[-2]
            self.ghostE[-1]   = self.ghostE[-2]
            self.ghostRho[-1] = self.ghostRho[-2]
        elif self.bc_upper == 'extrapolated':
            self.ghostU[-1]   = 2 * self.ghostU[-2] - self.ghostU[-3]
            self.ghostE[-1]   = 2 * self.ghostE[-2] - self.ghostE[-3]
            self.ghostRho[-1] = 2 * self.ghostRho[-2] - self.ghostRho[-3]


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
        self.ghostU[0]   = -self.ghostU[1]
        self.ghostE[0]   = self.ghostE[1]
        self.ghostRho[0] = self.ghostRho[1]
        ## transmissive at end
        self.ghostRho[-1] = self.ghostRho[-2]
        self.ghostU[-1]   = self.ghostU[-2]
        self.ghostE[-1]   = self.ghostE[-2]

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


class SedovBlast:
    def __init__(self,
                 ScaleLen__m: float,        # length scale
                 DomainLen__m: float,       # size of the domain
                 RExpl__m: float,           # radius of explosion
                 PExpl__Pa: float,          # pressure of explosion
                 tFinal__s: float,          # final simulation time
                 rho0__kgpm3: float=1.225,  # ambient air density, kg/m^3
                 P0__Pa: float=101325,      # ambient air pressure, Pa
                 order: int=0,              # order of the equations, 0=cartesian, 1-cylindrical, 2=spherical
                 gamma: float=1.4,          # ratio of specific heats, N/A
                 minNGridPts: int=500,      # minimum number of grid points
                 ):
        """
        Convert the parameters of the Sedov Blast to nondimensional form, for speed and numerical stability.
        """
        self.ScaleLen__m    = ScaleLen__m
        self.DomainLen__m   = DomainLen__m
        self.RExpl__m       = RExpl__m
        self.PExpl__Pa      = PExpl__Pa
        self.P0__Pa         = P0__Pa
        self.rho0__kgpm3    = rho0__kgpm3
        self.order          = order
        self.gamma          = gamma

        ## dimensionless parameters: scale using rho0, P0, and diameter
        UScale  = np.sqrt(P0__Pa / rho0__kgpm3)
        lenStar = DomainLen__m / ScaleLen__m
        rExpStar = RExpl__m / ScaleLen__m
        pExpStar = PExpl__Pa / P0__Pa
        tFinStar = tFinal__s * UScale / ScaleLen__m

        ## set up the radial grid, we want at least 10 points for the explosion
        nGridPts = lenStar / rExpStar * 10
        self.nGridPts = int( np.ceil( max(minNGridPts, nGridPts) ) )

        rMinStar = min(rExpStar / 10, lenStar / 100)
        self.grid = np.linspace(rMinStar, lenStar, num=self.nGridPts)
        self.times = np.linspace(0, tFinStar, num=minNGridPts)

        self.dr = self.grid[1] - self.grid[0]

        ## setting the initial conditions
        self.rho0 = np.ones_like(self.grid)
        self.p0   = np.ones_like(self.grid)
        self.v0   = np.zeros_like(self.grid)
        self.p0[self.grid < rExpStar] *= pExpStar

        ## time and grid in dimensional/metric scale
        self.r__m = self.grid * ScaleLen__m
        self.t__s = self.times * ScaleLen__m / UScale
        self.T__s, self.R__m = np.meshgrid(self.t__s, self.r__m)
    
    def solve(self,
              method: str='RK45'):
        """ Solve the system of partial differential equations using scipy.integrate.solve_ivp"""
        ODEs = EulerSol(self.grid, order=self.order, gamma=self.gamma,
                        alpha=[0.5, 0.5], beta=[0.25, 0.5])
        y0 = ODEs.createICs(self.rho0, self.v0, self.p0)
        t_range = [self.times.min(), self.times.max()]
        r_range = [self.grid.min(), self.grid.max()]
        
        print(f"Solving the Euler Equation as a system of ODES. \nt_range={t_range}(dimensionless) \nnGridPts={self.nGridPts}\nr_range={r_range}(dimensionless)")
        res = solve_ivp(ODEs, t_range, y0,
                        t_eval=self.times,
                        method=method)
        
        rhoStar_t, uStar_t, eStar_t, pStar_t = ODEs.conv2Primatives(res.y)
        self.rho = rhoStar_t * self.rho0__kgpm3
        self.u   = uStar_t * np.sqrt(self.P0__Pa / self.rho0__kgpm3)
        self.p   = pStar_t * self.P0__Pa
        self.E   = self.p / (self.rho * (self.gamma - 1)) + 0.5 * self.u**2


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
    LenScale__m = 1    # length scale of the problem
    DomainLen__m = 10   # size of the domain
    PAmb__Pa = 101325   # ambient air pressure
    PExpl__Pa = 20*PAmb__Pa # Explosive pressure
    RExpl__m = 3        # radius of explosion
    tFin__s  = 0.010    # final simulation time
    rhoAmb__kgpm3=1.225 # ambient air density
    orders = 2          # order of solution

    Blast = SedovBlast(LenScale__m, DomainLen__m, RExpl__m, PExpl__Pa, tFin__s,
                    P0__Pa=PAmb__Pa, rho0__kgpm3=rhoAmb__kgpm3, order=orders)
    Blast.solve()
    #Blast.dispFields()
    #Blast.plotDiscTimes()
    
    ## solving it outside SedovBlast - All dimensionless
    ## define grid
    rMin__m = 0.1
    tMax__s = 3
    nGridPts = 500
    rGrid = np.linspace(rMin__m, DomainLen__m, num=nGridPts)
    tGrid = np.linspace(0, tMax__s, num=nGridPts)

    ## define initial conditions
    P0__Pa      = 1
    PExpl__Pa   = 20 * P0__Pa
    rExpl__m    = 3
    rho0__kgpm3 = 1
    
    rho0 = rho0__kgpm3 * np.ones_like(rGrid)
    p0   = P0__Pa * np.ones_like(rGrid)
    p0[rGrid < rExpl__m] = PExpl__Pa
    vr0 = np.zeros_like(rGrid)

    ## set up system of equations
    ES = EulerSol(rGrid, order=orders)
    y0 = ES.createICs(rho0, vr0, p0)

    res = solve_ivp(ES, [tGrid.min(), tGrid.max()], y0, 
                    method='RK45',
                    t_eval=tGrid, )
    ## extracting primatives
    rho_t, U_t, E_t, p_t = ES.conv2Primatives(res.y)

    #%% density plots
    extent = [tGrid.min(), tGrid.max(), rGrid.min(), rGrid.max()]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    def densPlot(ax, var, desc, log=False):
        ## create a density plot of a given variable
        if log:
            cset = ax.imshow(np.flipud(np.log(var)), extent=extent, aspect='auto')
        else:
            cset = ax.imshow(np.flipud(var), extent=extent, aspect='auto')
        ax.set_title(desc)
        fig.colorbar(cset, ax=ax)
    
    densPlot(axes[0][0], rho_t, 'density (kg/m^3)')
    densPlot(axes[0][1], U_t,   'radial velocity (m/s)')
    densPlot(axes[1][0], E_t,   'total energy (J)')
    densPlot(axes[1][1], p_t,   'pressure (Pa)')
    axes[1][0].set_xlabel('time (ms)')
    axes[1][1].set_xlabel('time (ms)')
    axes[0][0].set_ylabel('r (m)')
    axes[1][0].set_ylabel('r (m)')

    #%% plots at discrete times
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

    linPlot(axes[0][0], rho_t, 'density (kg/m^3)')
    linPlot(axes[0][1], U_t,   'radial velocity (m/s)')
    linPlot(axes[1][0], E_t,   'total energy (J)', log=True)
    linPlot(axes[1][1], p_t,   'pressure (Pa)', log=True)
    axes[1][0].set_xlabel('r (m)')
    axes[1][1].set_xlabel('r (m)')
    axes[1][0].legend()