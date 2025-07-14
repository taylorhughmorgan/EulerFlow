/*
@author: Hugh Morgan
@date: 2025-07-09
@description: Solve the Taylor-Von Neumann-Sedov analytical solution to the Euler Equations using the self-similarity variable approach.
*/
#ifndef TAYLORNEUMANNSEDOV_H
#define TAYLORNEUMANNSEDOV_H

#include "mathutils.hpp"
#include <vector>

// state variables for self-similar solution
typedef struct {
    double Z, V, G;
} SelfSimilarState;

class TaylorSol
{
    // Taylor-Von-Neumann-Sedov solution
    public:
    size_t npts;
    double gamma;
    double rho0_kgpm3;
    double press0_Pa;
    double mu_Pas;
    std::vector<double> xi_arr;
    std::vector<SelfSimilarState> sols, res;
    TaylorSol(double m_rho0_kgpm3, double m_press0_Pa, size_t m_npts, double m_gamma, double m_mu_Pas);
};

#endif