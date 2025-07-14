/*
@author: Hugh Morgan
@date: 2025-07-09
@description: Solve the Taylor-Von Neumann-Sedov analytical solution to the Euler Equations using the self-similarity variable approach.
*/
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include "taylorneumannsedov.hpp"
#include "TVNS_coefs.h"

class SelfSimilarSol
{
    public:
    double gamma;
    double xi;
    double nu[5];
    SelfSimilarSol(double m_gam, double m_xi) {
        // initialize SelfSimilarClass
        gamma = m_gam;
        xi = m_xi;
        nu[0] = -1.0 * (13.0 * gamma*gamma - 7.0 * gamma + 12.0) / ((3.0 * gamma - 1.0) * (2.0 * gamma + 1.0));
        nu[1] = 5.0 * (gamma - 1.0) / (2.0 * gamma + 1.0);
        nu[2] = 3.0 / (2.0 * gamma + 1.0);
        nu[3] = -nu[0] / (2.0 - gamma);
        nu[4] = -2.0 / (2.0 - gamma);
    }
    double Z_rhs(double V) {
        //right-hand side of z
        return (gamma * (gamma - 1) * (1 - V) * V*V) / (2 * (gamma * V - 1.0));
    }
    double V_rhs(double V) {
        // right hand side of xi-V equation (xi as a function of V)
        double term1 = (gamma + 1.0) / (7.0 - gamma) * (5.0 - (3.0 * gamma - 1.0) * V);
        double term2 = (gamma + 1.0) / (gamma - 1.0) * (gamma * V - 1.0);
        return pow(0.5 * (gamma + 1.0) * V, -2) * pow(term1, nu[0]) * pow(term2, nu[1]);
    }
    double G_rhs(double V) {
        // right-hand side of G
        double term1 = (gamma + 1.0) / (7 - gamma) * (5.0 - (3.0 * gamma - 1.0) * V);
        double term2 = (gamma + 1.0) / (gamma - 1.0) * (gamma * V - 1.0);
        return (gamma + 1.0) / (gamma - 1.0) * pow(term2, nu[2]) * 
                pow(term1, nu[3]) * pow((gamma + 1.0) / (gamma - 1.0) * (1.0 - V), nu[4]);
    }
    void residual(SelfSimilarState X, SelfSimilarState * res) {
        // calculate the resiudal
        res->Z = X.Z - Z_rhs(X.V);
        res->V = xi - pow( V_rhs(X.V), 1.0/5.0);
        res->G = X.G - G_rhs(X.V);
    }
    double operator()(double V)
    {
        // objective function to minimize
        double xi_rhs = pow( V_rhs(V), 1.0/5.0 );
        return abs(xi_rhs - xi);
    }
};


TaylorSol::TaylorSol(double m_rho0_kgpm3, double m_press0_Pa, size_t m_npts, double m_gamma, double m_mu_Pas) 
{       
    // check for valid gammas
    if (m_gamma <= 1.0) {
        fprintf(stderr, "Invalid gamma: must be > 1.0\n");
        exit(EXIT_FAILURE);
    }
    // initialize Taylor-Von Neumann-Sedov Solution
    gamma = m_gamma;
    rho0_kgpm3 = m_rho0_kgpm3;
    press0_Pa = m_press0_Pa;
    npts = m_npts;
    mu_Pas = m_mu_Pas;
    // allocate arrays for xi, Z, G, and V
    xi_arr.resize(npts);
    sols.resize(npts);
    res.resize(npts);

    // populate xi_arr in reverse order, starting at 1
    double delta_xi = 1.0 / (double)npts;
    for (size_t i = 0; i < npts; ++i) 
        xi_arr[i] = 1.0 - delta_xi * i;

    // grab pre-processed values for V-solution and interpolate
    // find the gamma closest to the correct value
    size_t gamma_id = findClosest(GAMMAS, N_GAMMAS, gamma);
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_linear, npts);

    // reverse the coefficients to be linearly increasing
    double XI_ARR_FWD[N_PTS];
    double TVNS_COEFS_FWD[N_PTS];
    reverse_order(XI_ARR, N_PTS, XI_ARR_FWD);
    reverse_order(TVNS_COEFS[gamma_id], N_PTS, TVNS_COEFS_FWD);

    /*gsl_spline_init(spline, XI_ARR_FWD, TVNS_COEFS_FWD, npts);
    for (size_t i = 0; i < npts; ++i) {
        if (xi_arr[i] > XI_ARR_FWD[0] && xi_arr[i] < XI_ARR_FWD[N_PTS - 1]) {
            sols[i].V = gsl_spline_eval(spline, xi_arr[i], acc); 
        }
    }
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);*/
}