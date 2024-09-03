#!/usr/bin/env/python
"""
Author: Hugh Morgan
Date: 2024-09-02
Description: unit test on Sedov blast solver
"""
from numpy import all
from EulerFlow import SedovBlast

def test_sedov1d():
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
    Blast.dispFields()
    Blast.plotDiscTimes()
    assert all(Blast.p > 0) and Blast.p.max() < 2.0 * PExpl__Pa