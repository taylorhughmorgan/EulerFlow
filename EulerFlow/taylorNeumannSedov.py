#!/usr/bin/env python
"""
Author: Hugh Morgan
Date: 2024-08-26
Description: Solve the Taylor-Von Neumann-Sedov analytical solution to the Euler Equations using the self-similarity variable approach.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm

def ZVfunc(V, gamma):
    """ Z as a function of V """
    return (gamma * (gamma - 1) * (1 - V) * V**2) / (2 * (gamma * V - 1))

def GVfunc(V, gamma, nu):
    """ G as a function of V """
    common_term1 = (gamma + 1) / (7 - gamma) * (5 - (3*gamma - 1) * V)
    common_term2 = (gamma + 1) / (gamma - 1) * (gamma * V - 1)
    return (gamma + 1) / (gamma - 1) * common_term2**nu[2] * common_term1**nu[3] * ((gamma + 1) / (gamma - 1) * (1 - V))**nu[4]

def self_similar_sol(X, xi, gamma, nu):
    """ system of nonlinear equations that describe the self-similar solution to the Euler equations"""
    Z, V, G = X
    ## common terms that appears in both Xi and G right-hand-sides (RHS)
    common_term1 = (gamma + 1) / (7 - gamma) * (5 - (3*gamma - 1) * V)
    common_term2 = (gamma + 1) / (gamma - 1) * (gamma * V - 1)
    ## describe the RHS to Z, xi, and G equations
    rhsZ  = (gamma * (gamma - 1) * (1 - V) * V**2) / (2 * (gamma * V - 1))
    rhsV = (0.5 * (gamma + 1) * V)**-2 * common_term1**nu[0] * common_term2**nu[1]
    rhsG  = (gamma + 1)/(gamma - 1) * common_term2**nu[2] * common_term1**nu[3] * ((gamma + 1) / (gamma - 1) * (1 - V))**nu[4]
    return [Z - rhsZ, xi**(5./3.) - rhsV**(1./3.), G - rhsG]
        
def self_similar_Vxi(V, xi, gamma, nu):
    """ self-similar solution to calculate V as a function of xi """
    common_term1 = (gamma + 1) / (7 - gamma) * (5 - (3*gamma - 1) * V)
    common_term2 = (gamma + 1) / (gamma - 1) * (gamma * V - 1)
    rhsV = (0.5 * (gamma + 1) * V)**-2 * common_term1**nu[0] * common_term2**nu[1]
    return np.abs(xi - rhsV**(1./5.))

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
        self.gamma = gamma
        self.rho0__kgpm3 = rho0__kgpm3
        self.press0__Pa  = press0__Pa
        ## gamma-dependent coefficients to the sedov problem
        self.nus = np.zeros(5)
        self.nus[0] = -(13 * gamma**2 - 7 * gamma + 12) / ((3*gamma - 1) * (2*gamma + 1))
        self.nus[1] = 5 * (gamma - 1) / (2*gamma + 1)
        self.nus[2] = 3.0 / (2*gamma + 1)
        self.nus[3] = -self.nus[0] / (2 - gamma)
        self.nus[4] = -2.0 / (2 - gamma)
        
        ## looping through xi and solving the system of equation
        self.xi_arr = np.linspace(0, 1, num=npts)[::-1] ## looping backwards because the solution at xi=1 is known, use last solution as next guess
        self.sols   = np.zeros((npts,3))
        self.residuals = np.zeros_like(self.sols)
        initial_guess = 2 / (gamma + 1) #[1, 2/(gamma+1), 1]

        print("Solving self-similar solution...")
        for i, xi in tqdm(enumerate(self.xi_arr), total=npts):
            ## enforcing bounds on V to keep the solution real/stable
            res = minimize(self_similar_Vxi, initial_guess,
                           args=(xi, gamma, self.nus),
                           bounds=((1./gamma, 5 / (3 * gamma - 1)),),
                           method='L-BFGS-B', tol=1e-12,
                           )
            VTemp = res.x
            GTemp = GVfunc(VTemp, gamma, self.nus)
            ZTemp = ZVfunc(VTemp, gamma)
            if xi < 0.2:
                ZTemp = ZVfunc(VTemp + xi * 1e-10, gamma)
            tempSol = np.array([ZTemp, VTemp, GTemp]) #res.x
            self.sols[i,:] = tempSol.T
            self.residuals[i,:] = np.array(self_similar_sol(tempSol, xi, gamma, self.nus)).T
            ## use solution as next iteration's guess
            initial_guess = res.x
        
        self.Z, self.V, self.G = self.sols.T
        ## calculating residual error
        self.totreserror = np.linalg.norm(self.residuals, axis=1)
        self.toterror = np.sum(self.totreserror)
        
        ## interpolating Z, V, and G vs xi
        self.Z_xi = lambda xi: np.interp(xi, self.xi_arr[::-1], self.Z[::-1])
        self.V_xi = lambda xi: np.interp(xi, self.xi_arr[::-1], self.V[::-1])
        self.G_xi = lambda xi: np.interp(xi, self.xi_arr[::-1], self.G[::-1])

        ## calculating beta value, should be around 1.033 for gamma=1.4
        integrand = self.G * (self.V**2 / 2 + self.Z / (gamma * (gamma - 1))) * self.xi_arr**4
        rhs = -np.trapz(integrand, x=self.xi_arr)
        self.beta = (25 / (16 * np.pi * rhs))**(1./5.)
        print(f"For gamma={gamma:.1f}, beta={self.beta}, total error={self.toterror:.2f}")

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

        ## looping through each time and converting o primatives
        for it, t in enumerate(self.tGrid):
            ## find the shock location at the given time
            rShock = R_t(t)
            isWithinShock = self.rGrid < rShock
            rValsInShock = self.rGrid[isWithinShock]
            ## calculating density, pressure, and velocity in shock region by converting r to xi
            xi = rValsInShock / rShock
            rho = rho0__kgpm3 * self.G_xi(xi)
            self.rho[isWithinShock,it] = rho
            self.v[isWithinShock,it]   = (2 * rValsInShock / (5*t)) * self.V_xi(xi)
            self.p[isWithinShock,it]  += (rho / gamma) * self.Z_xi(xi) * (2 * rValsInShock / (5*t))**2 
        
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


    def plotDiscTimes(self, outevery=10):
        """ Plot each variable at discrete times """
        fig, axes = plt.subplots(nrows=2, ncols=2)
        tGrid__ms = self.tGrid * 1000
        for it, tTemp in enumerate(tGrid__ms):
            if it % outevery == 0:
                axes[0][0].plot(self.rGrid, self.rho[:,it], label=f"t={tTemp:.2f}ms")
                axes[1][0].plot(self.rGrid, self.v[:,it], label=f"t={tTemp:.2f}ms")
                axes[0][1].semilogy(self.rGrid, self.p[:,it], label=f"t={tTemp:.2f}ms")
                axes[1][1].semilogy(self.rGrid, self.E[:,it], label=f"t={tTemp:.2f}ms")
        
        axes[1][0].set_xlabel('distance (m)')
        axes[1][1].set_xlabel('distance (m)')
        axes[0][0].set_ylabel(r'density ($kg/m^3$)')
        axes[1][0].set_ylabel(r'velocity ($m/s$)')
        axes[0][1].set_ylabel(r'Pressure ($Pa$)')
        axes[1][1].set_ylabel(r'Total Energy ($J$)')
        for ax in axes.flatten():
            ax.grid(True)

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
        
        eachPlot(axes[0], self.Z, r"$Z(\xi)$", logPlot=True)
        eachPlot(axes[1], self.V, r"$V(\xi)$")
        eachPlot(axes[2], self.G, r"$G(\xi)$")
        eachPlot(axes[3], np.linalg.norm(self.residuals, axis=1), r"$residuals(\xi)$", logPlot=True)
        axes[3].set_xlabel(r'$\xi$')

    def plotScaledSol(self):
        """ Plot the scaled, self-similar solution p/p1, v/v1, rho/rho1 """
        ## scaling parameters: rho1, p1, v1
        rho1 = self.rho0__kgpm3 * (self.gamma + 1) / (self.gamma - 1)
        rho_over_rho1 = self.G * (self.gamma - 1) / (self.gamma + 1)
        v_over_v1 = self.xi_arr * self.V * (self.gamma + 1) / 2
        p_over_p1 = self.xi_arr**2 * self.G * self.Z * (self.gamma + 1) / (2 * self.gamma)
        T_over_T1 = self.xi_arr**2 * self.Z * (self.gamma + 1)**2 / (2 * self.gamma * (self.gamma - 1))

        fig, ax = plt.subplots()
        # plotting rho/rho_1, v/v_1, and p/p_1 on the same axis because they range from zero to 1
        ax.plot(self.xi_arr, rho_over_rho1, label=r'$\rho/\rho_1$')
        ax.plot(self.xi_arr, v_over_v1, label=r'$v/v_1$')
        ax.plot(self.xi_arr, p_over_p1, label=r'$p/p_1$')
        ax.set_xlabel(r"$\xi$")
        ax.grid(True)
        ax.legend(loc='center left')

        ## plotting temperature on another axis
        ax2 = ax.twinx()
        ax2.plot(self.xi_arr, T_over_T1, 'r', label=r'$T/T_1$')
        ax2.legend(loc='upper center')
        ax2.set_ylim([0, 1000])

if __name__ == '__main__':
    Eblast__J  = 1e10   ## blast energy
    rDomain__m = 20     ## domain of the problem

    TS = TaylorSol(Eblast__J, rDomain__m)
    TS.plotSelfSimilar()
    #TS.dispFields()
    #TS.plotDiscTimes()
    TS.plotScaledSol()