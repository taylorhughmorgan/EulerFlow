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

class particle_ode_1dr:
    """ system of equations for propagating a particle in a Sedov-Von Neumann-Taylor blast wave in 1D radial coordinates. """
    def __init__(self, FlowState: TaylorSol):
        self.FlowState = FlowState

    def __call__(self, t, y, *args, **kwds):
        """ system of equations: d(x_r)/dt = v_r """
        return self.FlowState.vrt_func(t, y)
    
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
    tRange = [0, TS.tGrid[-1]]
    r0 = np.array([10.0])
    part = particle_ode_1dr(TS)
    res = solve_ivp(part, tRange, r0, t_eval=TS.tGrid)
    
    ## calculating velocity
    vr = np.zeros_like(TS.tGrid)
    for i, t in enumerate(TS.tGrid):
        vr[i] = TS.vrt_func(t, res.y[0,i])

    ## plotting distance and velocity vs time
    fig, ax = plt.subplots()
    ax.plot(1000 * res.t, res.y.T)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('displacement (m)')
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.plot(1000 * res.t, vr, 'r')
    ax2.set_ylabel('velocity (m/s)')

# %%
