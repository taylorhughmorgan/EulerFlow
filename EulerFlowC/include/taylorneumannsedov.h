/*
@author: Hugh Morgan
@date: 2025-07-09
@description: Solve the Taylor-Von Neumann-Sedov analytical solution to the Euler Equations using the self-similarity variable approach.
*/
#ifndef TAYLORNEUMANNSEDOV_H
#define TAYLORNEUMANNSEDOV_H

#include <stdio.h>
#include <gsl/gsl_block.h>


// define self-similar function
typedef struct SelfSimilar
{
    /* Right hand side (RHS) of self-similar solution to the Sedov Von-Nuemann Taylor solution to the Euler eqns */
    double gamma;           // ratio of specific heats
    double xi;              // self-similar variable
    double nu[5];           // nu parameters
    double (*V_rhs)(struct SelfSimilar *, double);   // right-hand side of V-function
    double (*G_rhs)(struct SelfSimilar *, double);   // right-hand side of G-function
    double (*Z_rhs)(struct SelfSimilar *, double);   // right-hand side of Z-function
    void (*residual)(struct SelfSimilar *, double[3], double*); // residual
} SelfSimilarSol;

typedef struct 
{
    // Taylor-Von-Neumann-Sedov solution
    size_t npts;
    double gamma;
    double rho0_kgpm3;
    double press0_Pa;
    double mu_Pas;
    gsl_block *xi_arr, *Z_arr, *V_arr, *G_arr;
    double sols[3];
    double residuals[3];
} TaylorSol;

TaylorSol * init_TaylorSol(double rho0_kgpm3, double press0_Pa, size_t npts, double gamma, double mu_Pas);
void free_TaylorSol(TaylorSol * self);


// minimization function
int minimize(double guess, double lower_bound, double upper_bound, size_t max_iter, SelfSimilarSol * self);

#endif