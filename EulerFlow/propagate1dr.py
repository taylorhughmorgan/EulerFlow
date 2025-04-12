#!/usr/bin/env python
"""
Author: Hugh Morgan
Date: 2024-08-26
Description: propagate a small particle in a Sedov-Von Neumann-Taylor blast wave.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from EulerFlow import TaylorSol

class particle_ode_1ds:
    """ system of equations for propagating a particle in a given flow 
    state in 1D spherically-symmetric coordinates. """
    def __init__(self, FlowState: TaylorSol):
        self.FlowState = FlowState

    def __call__(self, t, y, *args, **kwds):
        """ system of equations: d(x_r)/dt = v_r and d(v_r)/dt = F_d """
        rho, press, vr = self.FlowState.flow_funcs(t, y)
        return vr

class PropagateParticle1ds:
    """ Propagate particles in 1D spherical coordinates"""
    def __init__(self, FlowState: TaylorSol):
        self.FlowState = FlowState
        self.__solution_reached = False

    def solve(self, r0, tFinal):
        """ Simulate the particle in a given Flow State"""
        # time domain
        tRange = [0, tFinal]
        tSol = np.linspace(tRange[0], tRange[1], num=200)
        r0 = np.array([r0])
        # set up system of equations and solve
        part = particle_ode_1ds(TS)
        res = solve_ivp(part, tRange, r0, t_eval=tSol)

        # save the solution
        self.tSol = res.t
        self.t__ms = 1000 * res.t
        self.r__m  = res.y.T
        
        ## calculating velocity
        self.vr__mps = np.zeros_like(self.r__m)
        for i, t in enumerate(self.tSol):
            self.vr__mps[i] = TS.vrt_func(t, res.y[0,i])

        self.__solution_reached = True

    def plotTraj(self):
        if not self.__solution_reached:
            raise Exception("Solution has not been reached yet. Run PropagateParticles1ds.solve() first")
        ## plotting distance and velocity vs time
        fig, ax = plt.subplots()
        ax.plot(self.t__ms, self.r__m)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('displacement (m)')
        ax.grid(True)

        ax2 = ax.twinx()
        ax2.plot(self.t__ms, self.vr__mps, 'r')
        ax2.set_ylabel('velocity (m/s)')
        ax2.tick_params(axis='y', colors='r')


if __name__ == '__main__':
    Eblast__J  = 1e10   ## blast energy
    rDomain__m = 20     ## domain of the problem
    #%% resolving the blast flow field
    TS = TaylorSol(Eblast__J, rDomain__m, 
                   time_interval='quadratic', method='TNC')
    TS.plotSelfSimilar()
    TS.dispFields()
    TS.plotDiscTimes()
    TS.plotScaledSol()

    #%% simulate the particle propagation in the blast wave
    traj = PropagateParticle1ds(TS)
    traj.solve(10, 2*TS.tFinal)
    traj.plotTraj()

# %%
