#!/usr/bin/env python
"""
@author: Hugh Morgan
@date: 2025-04-14
@description: symbolically solve the equations for the taylor-von neumann-sedov solution.
"""
import sympy as sp

# define symbols
r, t0, t, tau, mu, beta, E, rho_0 = sp.symbols('r t_0 t tau mu beta E rho_0',
                                             real=True, positive=True)
xi = sp.Function('xi')(r, t)
V_xi = sp.Function('V')(xi)
rho_f = sp.Function('rho_f')(r, t)

# define shock location R(t)
R_t = beta * (E * t**2 / rho_0 ) ** (sp.Rational(1,5))
# define xi(t)
xi_expr = r / R_t

# Now substitute this into u_f
u_f = (2 * r * V_xi) / (5 * t)

# differentiate wrt t
duf_dt = sp.diff(u_f, t)
dxi_dt = sp.diff(xi_expr, t)

# substitute dxi/dt with solution to d(xi)/dt
duf_dt_simp = duf_dt.subs(sp.diff(xi, t), dxi_dt)
# substitute xi back into expression and simplify
duf_dt_simp = duf_dt_simp.subs(xi_expr, xi)
duf_dt_simp = duf_dt_simp.simplify()
# solution should be (-2*r/5*t) * ( V(xi) + 2/5 * xi * V'(xi) )

#%% solving the Basset force equation
U_f = sp.Function('U_f')(r, tau)
integrand = 1 / sp.sqrt(t - tau) * sp.diff(U_f, tau)
#integrand = 1 / sp.sqrt(t - tau) * duf_dt_simp.subs(t, tau)
F_basset = sp.integrate(integrand, (tau, t0, t) )
dFb_dtau = sp.diff(F_basset, tau)
# %%
