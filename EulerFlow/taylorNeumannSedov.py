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
import time

class selfSimilarSol:
    """ Right hand side (RHS) of self-similar solution to the Sedov Von-Nuemann Taylor solution to the Euler eqns"""
    def __init__(self, gamma, xi):
        self.gamma = gamma
        self.xi = xi
        ## gamma-dependent coefficients to the sedov problem
        self.nu = np.zeros(5)
        self.nu[0] = -(13 * gamma**2 - 7 * gamma + 12) / ((3*gamma - 1) * (2*gamma + 1))
        self.nu[1] = 5 * (gamma - 1) / (2*gamma + 1)
        self.nu[2] = 3.0 / (2*gamma + 1)
        self.nu[3] = -self.nu[0] / (2 - gamma)
        self.nu[4] = -2.0 / (2 - gamma)
    
    def Zrhs(self, V):
        # right hand side of Z (function of V)
        return (self.gamma * (self.gamma - 1) * (1 - V) * V**2) / (2 * (self.gamma * V - 1))
    
    def Grhs(self, V):
        # right hand side of G (function of V)
        term1 = (self.gamma + 1) / (7 - self.gamma) * (5 - (3 * self.gamma - 1) * V)
        term2 = (self.gamma + 1) / (self.gamma - 1) * (self.gamma * V - 1)
        return (self.gamma + 1) / (self.gamma - 1) * term2**self.nu[2] * \
            term1**self.nu[3] * ((self.gamma + 1) / (self.gamma - 1) * (1 - V))**self.nu[4]

    def Vrhs(self, V):
        # right hand side of xi-V equation (xi as a function of V)
        term1 = (self.gamma + 1) / (7 - self.gamma) * (5 - (3*self.gamma - 1) * V)
        term2 = (self.gamma + 1) / (self.gamma - 1) * (self.gamma * V - 1)
        return (0.5 * (self.gamma + 1) * V)**-2 * term1**self.nu[0] * term2**self.nu[1]

    def residual(self, X):
        # calculate the residuals by taking the difference between X, V, and G and their RHS
        Z, V, G = X
        return np.array([Z - self.Zrhs(V), self.xi - self.Vrhs(V)**(1./5.), G - self.Grhs(V)])
    
    def __call__(self, V):
        # objective function for calling minimization
        return np.abs( self.xi - self.Vrhs(V)**(1./5.) )


class TaylorSol:
    def __init__(self, 
                 EBlast: float,             # blast energy released, joules
                 rDomain: float,            # maximum radius of the domain, meters
                 rho0__kgpm3: float=1.225,  # ambient air density, kg/m^3
                 press0__Pa: float=101325,  # ambient air pressure, Pa
                 npts: int=500,             # number of spatial points to solve for
                 time_interval: str='quadratic',# time scaling of the problem, either 'linear' or 'quadratic'
                 gamma: float=1.4,          # ratio of specific heats
                 tStart: float=0.0,         # start time in seconds
                 method: str='TNC',    # solution method used
                 ):
        """ Sedov solution """
        self.gamma = gamma
        self.rho0__kgpm3 = rho0__kgpm3
        self.press0__Pa  = press0__Pa
        self.EBlast__J   = EBlast
        
        ## looping through xi and solving the system of equation
        self.xi_arr = np.linspace(0, 1, num=npts)[::-1] ## looping backwards because the solution at xi=1 is known, use last solution as next guess
        self.sols   = np.zeros((npts,3))
        self.residuals = np.zeros_like(self.sols)
        initial_guess = 2 / (gamma + 1) #[1, 2/(gamma+1), 1]

        print("Solving self-similar solution...")
        simfunc = selfSimilarSol(gamma, self.xi_arr[-1])
        start_time = time.perf_counter()
        for i, xi in tqdm(enumerate(self.xi_arr), total=npts):
            ## enforcing bounds on V to keep the solution real/stable
            simfunc.xi = xi
            res = minimize(simfunc, initial_guess,
                           bounds=((1./gamma, 5 / (3 * gamma - 1)),),
                           method=method, tol=1e-12,
                           )
            VTemp = res.x
            GTemp = simfunc.Grhs(VTemp)
            ZTemp = simfunc.Zrhs(VTemp)
            tempSol = np.array([ZTemp, VTemp, GTemp])
            self.sols[i,:] = tempSol.T
            self.residuals[i,:] = simfunc.residual(tempSol).T
            ## use solution as next iteration's guess
            initial_guess = res.x

        execution_time = time.perf_counter() - start_time
        print(f"self-similar solution reached for {npts} points in {execution_time:.2f}s using '{method}' method.")
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
        print(f"For gamma={gamma:.1f}, beta={self.beta:.4f}, total error={self.toterror:.2f}")

        ## calculating the time it takes for the shock to reach the end of the domain
        self.tFinal = np.sqrt( rho0__kgpm3 / EBlast * (rDomain / self.beta)**5 )
        print(f"For domain size ={rDomain:.2f}m, time-of-shock arrival = {1000*self.tFinal:.2f}ms")

        ## setting up spatial and temporal grid
        if time_interval == 'linear':
            self.tGrid = np.linspace(tStart, self.tFinal, num=npts)
        elif time_interval == 'quadratic':
            self.tGrid = np.linspace(np.sqrt(tStart), np.sqrt(self.tFinal), num=npts)**2
            # sometimes the final value in tGrid is greater than the final time
            self.tGrid[self.tGrid > self.tFinal] = self.tFinal
        else:
            raise Exception(f"'{time_interval}' is not an acceptable argument for time scaling.")
        
        self.rGrid = np.linspace(0, rDomain, num=npts)
        self.T, self.R = np.meshgrid(self.tGrid, self.rGrid)
        ## initializing primatives
        self.rho = np.ones_like(self.T) * rho0__kgpm3
        self.v   = np.zeros_like(self.T)
        self.p   = np.ones_like(self.T) * press0__Pa

        ## determining what variables are within the shock
        ShockLoc = self.R_t(self.T) # offsetting shock by incredibly small number to avoid overruns
        isInShock = self.R < ShockLoc
        xi = self.R * isInShock / ShockLoc

        ## converting xi to primatives
        rho = rho0__kgpm3 * self.G_xi(xi[isInShock])
        self.rho[isInShock] = rho
        self.v[isInShock]   = (2 * self.R[isInShock] / (5 * self.T[isInShock])) * self.V_xi(xi[isInShock])
        self.p[isInShock]  += (rho / gamma) * self.Z_xi(xi[isInShock]) * (2 * self.R[isInShock] / (5 * self.T[isInShock]))**2

        ## rho, v, and p will have nan values at R and T=0, set them to default values
        self.rho[np.isnan(self.rho)] = rho0__kgpm3
        self.v[np.isnan(self.v)] = 0.0
        self.p[np.isnan(self.p)] = press0__Pa
        self.E = self.p / (self.rho * (gamma - 1)) + 0.5 * self.v**2
        print("Primative variables calculated")


    def R_t(self, t: float):
        ## function determining shock wave position as a function of time
        return self.beta * (self.EBlast__J * t**2 / self.rho0__kgpm3)**(1./5.)
    
    def vrt_func(self, t, r):
        """ Return the flow velocity as a function of radial distance and time """
        rShockFront = self.R_t(t)
        #vr = np.zeros_like(r)
        #xi = r / rShockFront
        #vr[r < rShockFront] = (2 * r / (5 * t)) * self.V_xi(xi)
        if r > rShockFront:
            return 0
        else:
            xi = r / rShockFront
            return (2 * r / (5 * t)) * self.V_xi(xi)
    
    def rhort_func(self, t, r):
        """ Return the flow density as a function of radial distance and time """
        rShockFront = self.R_t(t)
        if r > rShockFront:
            return self.rho0__kgpm3
        else:
            xi = r / rShockFront
            return self.rho0__kgpm3 * self.G_xi(xi)
        

    def prt_func(self, t, r):
        """ Return the flow pressure as a function of radial distance and time """
        rShockFront = self.R_t(t)
        if r > rShockFront:
            return self.press0__Pa
        else:
            xi = r / rShockFront
            rho = self.rhort_func(t, r)
            return (rho / self.gamma) * self.Z_xi(xi) * (2 * r / (5 * t))**2
    
    def flow_funcs(self, t, r):
        """ Return the flow density, pressure, and velocity as a function of radial distance and time """
        rShockFront = self.R_t(t)
        if r > rShockFront:
            rho = self.rho0__kgpm3
            press = self.press0__Pa
            vel = 0.0
        else:
            xi = r / rShockFront
            rho = self.rhort_func(t, r)
            press =  (rho / self.gamma) * self.Z_xi(xi) * (2 * r / (5 * t))**2
            vel = (2 * r / (5 * t)) * self.V_xi(xi)
        return np.array([rho, press, vel])
        
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
        ax.set_ylim([0,1])

        ## plotting temperature on another axis
        ax2 = ax.twinx()
        ax2.plot(self.xi_arr, T_over_T1, 'r', label=r'$T/T_1$')
        ax2.legend(loc='upper center')
        ax2.set_ylim([0, 1000])
        ax2.tick_params(axis='y', colors='r')
        ax2.set_ylim([0,1000])


if __name__ == '__main__':
    #%%
    Eblast__J  = 1e10   ## blast energy
    rDomain__m = 20     ## domain of the problem

    TS = TaylorSol(Eblast__J, rDomain__m, 
                   time_interval='quadratic', method='TNC')
    TS.plotSelfSimilar()
    TS.dispFields()
    TS.plotDiscTimes()
    TS.plotScaledSol()
# %%
