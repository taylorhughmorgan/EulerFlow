#!/usr/bin/env/python
"""
Author: Hugh Morgan
Date: 2024-09-02
Description: unit test on Sedov blast solver
"""
from numpy import all
from EulerFlow import SedovBlast

def test_sedov1d():
    LenScale__m = 1         # length scale of the problem
    DomainLen__m = 10       # size of the domain
    PAmb__Pa = 101325       # ambient air pressure
    PExpl__Pa = 60*PAmb__Pa # Explosive pressure
    RExpl__m = 2.0          # radius of explosion
    tFin__s  = 0.010        # final simulation time
    rhoAmb__kgpm3=1.225     # ambient air density
    orders = 2              # order of solution

    Blast = SedovBlast(LenScale__m, DomainLen__m, RExpl__m, PExpl__Pa, tFin__s,
                    P0__Pa=PAmb__Pa, rho0__kgpm3=rhoAmb__kgpm3, order=orders)
    Blast.solve()
    Blast.plotDiscTimes()
    Blast.dispFields()
    assert all(Blast.p > 0) and Blast.p.max() < 2.0 * PExpl__Pa

def test_weakShock():
    LenScale__m = 1         # length scale of the problem
    DomainLen__m = 20       # size of the domain
    PAmb__Pa = 101325       # ambient air pressure
    PExpl__Pa = 10*PAmb__Pa # Explosive pressure
    RExpl__m = 2.0          # radius of explosion
    tFin__s  = 0.04         # final simulation time
    rhoAmb__kgpm3=1.225     # ambient air density
    orders = 2              # order of solution

    Blast = SedovBlast(LenScale__m, DomainLen__m, RExpl__m, PExpl__Pa, tFin__s,
                    P0__Pa=PAmb__Pa, rho0__kgpm3=rhoAmb__kgpm3, order=orders)
    Blast.solve()
    Blast.plotDiscTimes()
    Blast.dispFields()
    assert all(Blast.p > 0) and Blast.p.max() < 2.0 * PExpl__Pa

if __name__ == '__main__':
    test_sedov1d()
    #test_weakShock()