# EulerFlow
## Description
Solve the Euler (Compressible, inviscid Navier Stokes) equations in cartesian, cylindrical, and spherical coordinates in python. Validation is done through the self-similar to the solution, described as the Taylor-Von Neumann-Sedov solution.

## Numerical Methods for the Euler Solution

## Example
### Import Libraries
For this simple example, we're going to use `SedovBlast`, which handles setting up the dimensionless problem. Using SI units to solve the Euler equations slows down the problem significanly and leads to numerical instability. It utilizes the Jameson-Shmidt-Turkel finite volume scheme for spatial discretization and `scipy.integrate.solve_ivp` for the time integration. The boundary conditions at the origin are reflective and transmissive at the exit. Unit tests have shown that the wave transmits easily through the end and that the wave reflects at the origin.
```python
from EulerFlow import SedovBlast
```

### Set Up Blast Scenario
```python
LenScale__m = 1    # length scale of the problem
DomainLen__m = 10   # size of the domain
PAmb__Pa = 101325   # ambient air pressure
PExpl__Pa = 20*PAmb__Pa # Explosive pressure
RExpl__m = 3        # radius of explosion
tFin__s  = 0.010    # final simulation time
rhoAmb__kgpm3=1.225 # ambient air density
orders = 2          # order of solution
```
We look at a blast solition over the domain 0 to 10 meters. We scale the parameters by 1 meter, but values 1-10 are acceptable. We set the ambient pressure and density to STP: `PAmb__Pa=101325` and `rhoAmb__kgpm3=1.225`. The Explosion is centered at the origin at t=0, and has a radius of 3 meters. We allow the solution to solve for 10 miliseconds, and set order=2 (spherical coordinates). order=0 indicates cartesian coordinates, and order=1 indicates cylindrical coordinates.

### Run Simulation
`SedovBlast` converts the SI coordinates, time, and thermodynamic parameters into dimensionless units. `SedovBlast.solve()` takes the argument of method, which is the time integration scheme for `scipy.integrate.solve_ivp`. The solution should converge in a few seconds.
```python
Blast = SedovBlast(LenScale__m, DomainLen__m, RExpl__m, PExpl__Pa, tFin__s,
                P0__Pa=PAmb__Pa, rho0__kgpm3=rhoAmb__kgpm3, order=orders)
Blast.solve(method='RK45')
```

### Analyze Results
We can then view the fields as density plots or at discrete times.
```python
Blast.dispFields()      ## density plots of the field
Blast.plotDiscTimes()   ## plott at discrete times
```
## Taylor-Von Neumann-Sedov Analytical Solution