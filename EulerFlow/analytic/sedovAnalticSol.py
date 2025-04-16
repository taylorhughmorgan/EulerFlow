#!/usr/bin/env python
"""
@author: Hugh Morgan
@date: 2025-04-14
@description: symbolically solve the equations for the taylor-von neumann-sedov solution.
"""
import sympy as sp

# define symbols
r, t, rho_f, mu, beta, E, rho_0 = sp.symbols('r t rho_f mu beta E rho_0',
                                             real=True, positive=True)
xi = sp.Function('xi')(r, t)
V_xi = sp.Function('V')(xi)

# define shock location R(t)
R_t = beta * (E * t**2 / rho_0 ) ** (sp.Rational(1,5))
# define xi(t)
xi_expr = r / R_t

# Now substitute this into u_f
u_f = (2 * r * V_xi) / (5 * t)

# differentiate wrt t
duf_dt = sp.diff(u_f, t)
dxi_dt = sp.diff(xi_expr, t)