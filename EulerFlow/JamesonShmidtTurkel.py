#!/usr/bin/env python
"""
Author: Hugh Morgan
Date: 2024-08-26
Description: Solve the 1D Euler equations in cartesian, cylindrical, and polar coordinates using the Jameson-Shmidt Turkel numerical scheme in 1D.
"""
import numpy as np

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