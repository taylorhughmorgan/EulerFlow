/*
@author: Hugh Morgan
@date: 2025-07-09
@description: test the analytical solution to the taylor-Von Neumann-Sedov solution
*/
#include "taylorneumannsedov.hpp"
#include "TVNS_coefs.h"

int main() {
    double rho0_kgpm3 = 1.225;
    double p0_Pa = 101325.0;
    size_t npts = 10;
    double gamma = 1.4;
    double mu_Pas = 1.789e-5;
    // create taylor solution object
    TaylorSol TS(rho0_kgpm3, p0_Pa, npts, gamma, mu_Pas);
    return 0;
}