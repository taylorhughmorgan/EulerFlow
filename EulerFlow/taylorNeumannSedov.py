#!/usr/bin/env python
"""
Author: Hugh Morgan
Date: 2024-08-26
Description: Solve the Taylor-Von Neumann-Sedov analytical solution to the Euler Equations using the self-similarity variable approach.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from tqdm import tqdm

class TaylorSol:
    def __init__(self, 
                 EBlast: float,             # blast energy released, joules
                 rDomain: float,            # maximum radius of the domain, meters
                 rho0__kgpm3: float=1.225,  # ambient air density, kg/m^3
                 press0__Pa: float=101325,  # ambient air pressure, Pa
                 npts: int=500,             # number of spatial points to solve for
                 time_interval: str='quadratic',
                 gamma: float=1.4,
                 ):
        """ Sedov solution """
        ## gamma-dependent coefficients to the sedov problem
        nu1 = -(13 * gamma**2 - 7 * gamma + 12) / ((3*gamma - 1) * (2*gamma + 1))
        nu2 = 5 * (gamma - 1) / (2*gamma + 1)
        nu3 = 3.0 / (2*gamma + 1)
        nu4 = -nu1 / (2 - gamma)
        nu5 = -2.0 / (2 - gamma)

        def func(x, xi=0.5):
            """ system of nonlinear equations that describe the self-similar solution to the Euler equations"""
            Z, V, G = x
            ## common terms that appears in both Xi and G right-hand-sides (RHS)
            common_term1 = (gamma + 1) / (7 - gamma) * (5 - (3*gamma - 1) * V)
            common_term2 = (gamma + 1) / (gamma - 1) * (gamma * V - 1)
            ## describe the RHS to Z, xi, and G equations
            rhsZ  = (gamma * (gamma - 1) * (1 - V) * V**2) / (2 * (gamma * V - 1))
            rhsXi = (0.5 * (gamma + 1) * V)**-2 * common_term1**nu1 * common_term2**nu2
            rhsG  = (gamma + 1)/(gamma - 1) * common_term2**nu3 * common_term1**nu4 * ((gamma + 1) / (gamma - 1) * (1 - V))**nu5
            return [Z - rhsZ, xi**5 - rhsXi, G - rhsG]
        
        ## looping through xi and solving the system of equation
        self.xi_arr = np.linspace(0, 1, num=npts)[::-1] ## looping backwards because the solution at xi=1 is known, use last solution as next guess
        self.sols   = np.zeros((npts,3))
        self.residuals = np.zeros_like(self.sols)
        initial_guess = [1, 2/(gamma+1), 1]

        print("Solving self-similar solution...")
        for i, xi in tqdm(enumerate(self.xi_arr), total=npts):
            ## enforcing bounds on V to keep the solution real/stable
            res = least_squares(func, initial_guess, args=(xi,),
                                bounds=((-np.inf, 1./gamma, -np.inf), (np.inf, 5/(3*gamma - 1), np.inf))
                                )
            self.sols[i,:] = res.x
            self.residuals[i,:] = func(res.x, xi=xi)
            ## use solution as next iteration's guess
            initial_guess = res.x
        
        self.Z, self.V, self.G = self.sols.T
        ## interpolating Z, V, and G vs xi
        Z_xi = lambda xi: np.interp(xi, self.xi_arr[::-1], self.Z[::-1])
        V_xi = lambda xi: np.interp(xi, self.xi_arr[::-1], self.V[::-1])
        G_xi = lambda xi: np.interp(xi, self.xi_arr[::-1], self.G[::-1])

        ## calculating beta value, should be around 1.033 for gamma=1.4
        integrand = self.G * (self.V**2 / 2 + self.Z / (gamma * (gamma - 1))) * self.xi_arr**4
        rhs = -np.trapz(integrand, x=self.xi_arr)
        self.beta = (25 / (16 * np.pi * rhs))**(1./5.)
        print(f"For gamma={gamma:.1f}, beta={self.beta}")

        ## converting to primative variables
        def R_t(t: float):
            ## function determining shock wave position as a function of time
            return self.beta * (EBlast * t**2 / rho0__kgpm3)**(1./5.)
        
        ## calculating the time it takes for the shock to reach the end of the domain
        self.tFinal = np.sqrt( rho0__kgpm3 / EBlast * (rDomain / self.beta)**5 )
        print(f"For domain size ={rDomain:.2f}m, time-of-shock arrival = {1000*self.tFinal:.2f}ms")

        ## setting up spatial and temporal grid
        self.tGrid = np.linspace(0, self.tFinal, num=2*npts)
        self.rGrid = np.linspace(0, rDomain, num=2*npts)
        self.T, self.R = np.meshgrid(self.tGrid, self.rGrid)
        ## initializing primatives
        self.rho = np.ones_like(self.T) * rho0__kgpm3
        self.v   = np.zeros_like(self.T)
        self.p   = np.ones_like(self.T) * press0__Pa

        ## looping through each time and converting to primatives
        for it, t in enumerate(self.tGrid):
            ## find the shock location at the given time
            rShock = R_t(t)
            isWithinShock = self.rGrid < rShock
            rValsInShock = self.rGrid[isWithinShock]
            ## calculating density, pressure, and velocity in shock region by converting r to xi
            xi = rValsInShock / rShock
            rho = rho0__kgpm3 * G_xi(xi)
            self.rho[isWithinShock,it] = rho
            self.v[isWithinShock,it]   = (2 * rValsInShock / (5*t)) * V_xi(xi)
            self.p[isWithinShock,it]  += (rho / gamma) * Z_xi(xi) * (2 * rValsInShock / (5*t))**2 
        
        self.E = self.p / (self.rho * (gamma - 1)) + 0.5 * self.v**2
        print("Primative variables calculated")


    def dispFields(self):
        """ Display the field variables as functions of space and time """
        fig, axes = plt.subplots(nrows=2, ncols=2)
        def field(ax, val, desc, logPlot=False):
            if logPlot: norm='log'
            else: norm='linear'
            cs = ax.pcolormesh(1000*self.T, self.R, val,
                           norm=norm, cmap='jet')
            fig.colorbar(cs, ax=ax, label=desc)

        field(axes[0][0], self.rho, r"density ($kg/m^3$)")
        field(axes[1][0], self.v,   r"velocity ($m/s$)", logPlot=True)
        field(axes[0][1], self.p,   r"Pressure ($Pa$)", logPlot=True)
        field(axes[1][1], self.E,   r"Total Energy ($J$)", logPlot=True)
        axes[1][0].set_xlabel('time (ms)')
        axes[1][1].set_xlabel('time (ms)')
        axes[0][0].set_ylabel('distance (m)')
        axes[1][0].set_ylabel('distance (m)')


    def plotSelfSimilar(self):
        """ Plot results of self-similar solution to the Taylor-Von Neumann-Sedov blast problem"""
        fig, axes = plt.subplots(nrows=4)
        def eachPlot(ax, val, desc, logPlot=False):
            if logPlot:
                ax.semilogy(self.xi_arr, val)
            else:
                ax.plot(self.xi_arr, val)
            ax.set_ylabel(desc)
            ax.grid(True)
        
        eachPlot(axes[0], self.Z, r"$Z(\xi)$")
        eachPlot(axes[1], self.V, r"$V(\xi)$")
        eachPlot(axes[2], self.G, r"$G(\xi)$")
        eachPlot(axes[3], np.linalg.norm(self.residuals, axis=1), r"$residuals(\xi)$", logPlot=True)


if __name__ == '__main__':
    Eblast__J  = 1e8    ## blast energy
    rDomain__m = 20     ## domain of the problem

    TS = TaylorSol(Eblast__J, rDomain__m)
    TS.plotSelfSimilar()
    TS.dispFields()