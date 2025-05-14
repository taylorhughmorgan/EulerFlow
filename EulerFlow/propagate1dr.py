#!/usr/bin/env python
"""
@author: Hugh Morgan
@date: 2025-04-09
@description: propagate a small particle in a Sedov-Von Neumann-Taylor blast wave.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from EulerFlow import TaylorSol

class ODEBase:
    """ Base class for ODE systems for propagators. Assigns particle and flow properties. """
    def __init__(self, 
                 FlowState: TaylorSol,  # flow state 
                 mass__kg: float,       # particle mass
                 Aref__m2: float,       # particle ballistic reference area
                 Cd__: float,           # particle drag coefficient
                 Vol__m3: float,        # particle volume
                 ):
        self.FlowState = FlowState
        self.mass = mass__kg
        self.Aref = Aref__m2
        self.Cd__ = Cd__
        self.Vol  = Vol__m3


class drag_ode_1ds(ODEBase):
    """ system of equations for drag-propagating a particle in a given flow 
    state in 1D spherically-symmetric coordinates. """
    def __call__(self, t, y, *args, **kwds):
        """ system of equations: 
            d(x_r)/dt = v_r 
            d(v_r)/dt = F_d / m 
        """
        r, v_r = y
        # calculate the flow state
        rho_flow, press_flow, vr_flow = self.FlowState.flow_funcs(t, r)
        # calculate the reference velocity, aka diference between particle and flow velocity
        delta_vr = vr_flow - v_r
        # calculate drag force
        F_r = 0.5 * self.Cd__ * rho_flow * self.Aref * delta_vr * np.abs(delta_vr)
        a_r = F_r / self.mass
        return np.array([v_r, a_r])


class BBO_ode_1ds(ODEBase):
    """ System of equations for particle propagations using Basette-Bousinesque-Oseen
        model. """
    def __init__(self, *args, radius__m, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius__m
        self.diam   = 2 * radius__m

    def __call__(self, t, y, *args, **kwds):
        r, v_r = y

        # calculate the flow state
        rho_flow, press_flow, vr_flow = self.FlowState.flow_funcs(t, r)
        dvrdt_flow = self.FlowState.dvrdtrt_func(t, r)
        m_fluid = self.Vol * rho_flow # displaced fluid mass

        # calculate the reference velocity, aka diference between particle and flow velocity
        delta_vr = vr_flow - v_r
        ## calculate forces
        # drag force
        F_drag = 0.5 * self.Cd__ * rho_flow * self.Aref * delta_vr * np.abs(delta_vr)

        # Froude–Krylov force due to the pressure gradient in the undisturbed flow
        gradP = self.FlowState.gradP_func(t, r)
        F_fk  = -self.Vol * gradP

        # added mass force 
        F_am = 0.5 * m_fluid * dvrdt_flow

        # Basset force
        F_ba = 0.0

        # summing all forces
        F_net = F_drag + F_fk + F_am + F_ba
        a_r = F_net / (self.mass + 0.5 * m_fluid)

        return np.array([v_r, a_r])
    

class PropagateParticle1ds:
    """ Propagate particles in 1D spherical coordinates"""
    def __init__(self, FlowState: TaylorSol):
        self.FlowState = FlowState
        self.__solution_reached = False

    def solve(self, 
              r0: float,        # initial position
              v0: float,        # initial velocity
              tFinal: float,    # final time
              part_mass: float, # particle mass
              part_Aref: float, # particle reference area
              part_Cd: float,   # particle drag coefficient
              part_vol: float,  # particle volume
              propagator='drag',# use either 'drag' or 'BBO' for propagator
              ):
        """ Simulate the particle in a given Flow State"""
        # time domain
        tRange = [0, tFinal]
        tSol = np.linspace(tRange[0], tRange[1], num=200)
        r0 = np.array([r0, v0])

        # set up system of equations and solve
        if propagator == 'drag':
            part = drag_ode_1ds(self.FlowState, part_mass, part_Aref, part_Cd, part_vol)
        elif propagator == 'BBO':
            part_radius = (3 * part_vol / (4 * np.pi) ) ** (1.0/3.0)
            part = BBO_ode_1ds(self.FlowState, part_mass, part_Aref, part_Cd, part_vol, radius__m=part_radius)
        else:
            raise Exception(f"propgator='{propagator}' is not recognized. Try 'drag' or 'BBO'. ")
        
        res = solve_ivp(part, tRange, r0, t_eval=tSol)

        # save the solution
        self.tSol = res.t
        self.t__ms = 1000 * res.t
        self.r__m  = res.y[0,:]
        self.vr__mps = res.y[1,:]
        
        ## calculating flow velocity
        self.vrflow__mps = np.zeros_like(self.r__m)
        for i, t in enumerate(self.tSol):
            self.vrflow__mps[i] = TS.vrt_func(t, res.y[0,i])

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
        ax2.plot(self.t__ms, self.vr__mps, 'r', label='particle vel')
        ax2.plot(self.t__ms, self.vrflow__mps, 'r--', label='flow vel')
        ax2.set_ylabel('velocity (m/s)')
        ax2.tick_params(axis='y', colors='r')
        ax2.set_ylim([0, 1.1 * self.vr__mps.max()])


if __name__ == '__main__':
    Eblast__J  = 1e10   # blast energy
    rDomain__m = 20     # domain of the problem
    #%% resolving the blast flow field
    TS = TaylorSol(method='TNC')
    TS.solve(Eblast__J, rDomain__m, time_interval='quadratic')
    TS.plotSelfSimilar()
    TS.dispFields()
    TS.plotDiscTimes()
    TS.plotScaledSol()

    #%% simulate the particle propagation in the blast wave
    # assume 1kg cube of cork
    mass__kg    = 1.0       # particle mass
    rho__kgpm3  = 200.0     # particle density
    # calculating volume, radius, ballistic reference area
    vol__m3     = mass__kg / rho__kgpm3
    radius__m   = vol__m3 ** (1./3.)
    Aref__m2    = radius__m**2
    Cd__        = 0.75      # drag coefficient
    r0__m       = 10.0      # initial position
    vr0__mps    = 0.0       # initial velocity

    traj = PropagateParticle1ds(TS)
    traj.solve(r0__m, vr0__mps, 2*TS.tFinal, mass__kg, Aref__m2, Cd__, vol__m3,
               propagator='BBO')
    traj.plotTraj()

# %%
