#!/usr/bin/env python
"""
Author: Hugh Morgan
Date: 2024-08-26
Description: Solve the 1D Euler equations in cartesian, cylindrical, and polar coordinates using the Jameson-Shmidt Turkel numerical scheme in 1D.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from math import gamma as gamma_func
from EulerFlow.EulerSol1D import EulerSol, shockReachEnd
from EulerFlow.mathutils import hyperbolic_step


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
                 relaxFactor: float=0.15,   # step function relaxation factor, improves numerical stability
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

        ## calculate the internal energy of the explosion
        n = order + 1
        self.VExpl__m3 = np.pi ** (n/2) / gamma_func(n/2 + 1) * RExpl__m**n
        self.EExpl__J  = self.PExpl__Pa * self.VExpl__m3 / (self.gamma - 1)

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
        self.tStar = np.linspace(0, tFinStar, num=minNGridPts)

        self.dr = self.grid[1] - self.grid[0]

        ## setting the initial conditions
        self.rho0 = np.ones_like(self.grid)
        #self.p0   = np.ones_like(self.grid)
        self.v0   = np.zeros_like(self.grid)
        ## use hyperbolic tangent instead of step function for numerical stability
        self.p0   = pExpStar * hyperbolic_step(self.grid, rExpStar, delta=relaxFactor)
        self.p0[self.p0 < 1] = 1.0

        ## time and grid in dimensional/metric scale
        self.r__m = self.grid * ScaleLen__m
        self.UScale = UScale
    

    def solve(self,
              method: str='RK45',
              terminate: bool=False,    # terminate prematurely if the shock front reaches the end of the domain
              ):
        """ Solve the system of partial differential equations using scipy.integrate.solve_ivp"""
        ODEs = EulerSol(self.grid, order=self.order, gamma=self.gamma,
                        alpha=[0.5, 0.5], beta=[0.25, 0.5])
        y0 = ODEs.createICs(self.rho0, self.v0, self.p0)
        t_range = [self.tStar.min(), self.tStar.max()]
        r_range = [self.grid.min(), self.grid.max()]
        
        ## set stopping criteria
        stop_criteria = shockReachEnd(self.grid, self.order, self.gamma, terminate)
        print(f"Solving the Euler Equation as a system of ODES. \nt_range={t_range}(dimensionless) \nnGridPts={self.nGridPts}\nr_range={r_range}(dimensionless)")
        ## timing the execution
        start_time = time.perf_counter()
        res = solve_ivp(ODEs, t_range, y0,
                        t_eval=self.tStar,
                        method=method,
                        events=stop_criteria)

        ## reporting the run time
        execution_time = time.perf_counter() - start_time
        print(f"Run completed! Execution time = {1000*execution_time:.3f}ms")
        
        ## creating a time array based on result times (possible conflict with self.times as it can about prematurely)
        self.res = res
        self.t__s = self.ScaleLen__m / self.UScale * res.t
        self.T__s, self.R__m = np.meshgrid(self.t__s, self.r__m)

        ## converting results.y into primatives then converting to SI values
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
            for it, tTemp in enumerate(self.t__s):
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
    #%%
    LenScale__m = 1     # length scale of the problem
    DomainLen__m = 15   # size of the domain
    PAmb__Pa = 101325   # ambient air pressure
    PExpl__Pa = 80*PAmb__Pa # Explosive pressure
    RExpl__m = 1.0      # radius of explosion
    tFin__s  = 0.031    # final simulation time
    rhoAmb__kgpm3=1.225 # ambient air density
    orders = 2          # order of solution

    Blast = SedovBlast(LenScale__m, DomainLen__m, RExpl__m, PExpl__Pa, tFin__s,
                    P0__Pa=PAmb__Pa, rho0__kgpm3=rhoAmb__kgpm3, order=orders,
                    relaxFactor=0.15, minNGridPts=1000)
    Blast.solve(terminate=False)
    Blast.dispFields()
    Blast.plotDiscTimes()

    #%% compare to analytical solution
    from EulerFlow.taylorNeumannSedov import TaylorSol
    # calculating blast energy
    EBlast = (DomainLen__m / 1.033)**5 * (rhoAmb__kgpm3 / tFin__s**2)
    TS = TaylorSol(rho0__kgpm3=rhoAmb__kgpm3, press0__Pa=PAmb__Pa, npts=1000)
    TS.solve(Blast.EExpl__J, DomainLen__m)
    
    #%% plotting
    test_it = 900
    fig = plt.figure()
    plt.plot(Blast.r__m, Blast.u[:,test_it], label='numerical')
    plt.plot(TS.rGrid, TS.v[:,test_it], label='analytical')
    plt.xlabel('radial grid (m)')
    plt.ylabel('velocity (m/s)')
    plt.legend()
    plt.grid(True)
    
    #%% plot max pressure across times
    fig, ax = plt.subplots(nrows=2)
    ax[0].semilogy(Blast.r__m, Blast.p.max(axis=1), label='numerical')
    ax[0].semilogy(TS.rGrid, TS.p.max(axis=1), label='analytic')
    ax[0].set_ylabel('pressure (psi)')
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_ylim([PAmb__Pa, Blast.p.max()])
    ## plotting max velocity
    ax[1].plot(Blast.r__m, Blast.u.max(axis=1), label='numerical')
    ax[1].plot(TS.rGrid, TS.v.max(axis=1), label='analytic')
    ax[1].set_ylabel('velocity (m/s)')
    ax[1].set_xlabel('radial grid (m)')
    ax[1].grid(True)
    ax[1].set_ylim([0, Blast.u.max()])

# %%
