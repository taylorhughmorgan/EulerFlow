/*
@author: Hugh Morgan
@date: 2025-07-09
@description: Solve the Taylor-Von Neumann-Sedov analytical solution to the Euler Equations using the self-similarity variable approach.
*/
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_min.h>
#include "taylorneumannsedov.h"


double Z_func(SelfSimilarSol * self, double V) {
    // right-hand side of z
    return (self->gamma * (self->gamma - 1) * (1 - V) * V*V) / (2 * (self->gamma * V - 1));
}
double G_func(SelfSimilarSol * self, double V) {
    // right-hand side of G
    double term1 = (self->gamma + 1) / (7 - self->gamma) * (5 - (3 * self->gamma - 1) * V);
    double term2 = (self->gamma + 1) / (self->gamma - 1) * (self->gamma * V - 1);
    return (self->gamma + 1) / (self->gamma - 1) * pow(term2, self->nu[2]) * 
            pow(term1, self->nu[3]) * pow((self->gamma + 1) / (self->gamma - 1) * (1 - V), self->nu[4]);
}
double V_func(SelfSimilarSol * self, double V) {
    // right hand side of xi-V equation (xi as a function of V)
    double term1 = (self->gamma + 1) / (7 - self->gamma) * (5 - (3*self->gamma - 1) * V);
    double term2 = (self->gamma + 1) / (self->gamma - 1) * (self->gamma * V - 1);
    return pow(0.5 * (self->gamma + 1) * V, -2) * pow(term1, self->nu[0]) * pow(term2, self->nu[1]);
}

void residual(SelfSimilarSol * self, double X[3], double * res) {
    // calculate the residual
    double Z = X[0], V = X[1], G = X[2];
    res[0] = Z - self->Z_rhs(self, V);
    res[1] = self->xi - pow( self->V_rhs(self, V), 1.0/5.0);
    res[2] = G - self->G_rhs(self, V);
}


void init_SelfSimilarSol(SelfSimilarSol * self, double gamma, double xi) {
    //Right hand side (RHS) of self-similar solution to the Sedov Von-Nuemann Taylor solution to the Euler eqns
    self->gamma = gamma;
    self->xi = xi;
    self->nu[0] = -(13 * gamma*gamma - 7 * gamma + 12) / ((3 * gamma - 1) * (2 * gamma + 1));
    self->nu[1] = 5 * (gamma - 1) / (2 * gamma + 1);
    self->nu[2] = 3.0 / (2 * gamma + 1);
    self->nu[3] = -self->nu[0] / (2 - gamma);
    self->nu[4] = -2.0 / (2 - gamma);
    self->G_rhs = G_func;
    self->V_rhs = V_func;
    self->Z_rhs = Z_func;
    self->residual = residual;
}

double obj_function(double V, void * params)
{
    // objective function to minimize
    SelfSimilarSol self = *(SelfSimilarSol *)(params);
    return abs(self.xi - pow(self.V_rhs(&self, V), 1.0/5.0) );
    //(void)(params); /* avoid unused parameter warning */
    //return cos(V) + 1.0;
}


TaylorSol * init_TaylorSol(double rho0_kgpm3, double press0_Pa, size_t npts, double gamma, double mu_Pas)
{
    // initialize Taylor-Von Neumann-Sedov Solution
    TaylorSol * self = (TaylorSol *)malloc(sizeof(TaylorSol));
    self->gamma = gamma;
    self->rho0_kgpm3 = rho0_kgpm3;
    self->press0_Pa = press0_Pa;
    self->npts = npts;
    self->mu_Pas = mu_Pas;
    // allocate arrays for xi, Z, G, and V
    self->xi_arr = gsl_block_alloc(npts);
    self->V_arr = gsl_block_alloc(npts);
    self->G_arr = gsl_block_alloc(npts);
    self->Z_arr = gsl_block_alloc(npts);

    // populate xi_arr in reverse order, starting at 1
    double delta_xi = 1.0 / (double)npts;
    for (size_t i = 0; i < npts; ++i) 
        self->xi_arr->data[i] = 1.0 - delta_xi * i;
    
    // loop through xi and solve system of equations at each xi
    double initial_guess = 2.0 / (gamma + 1.0); // initial guess
    double lower_bound = 1.0 / gamma;
    double upper_bound = 5.0 / (3.0 * gamma - 1.0);

    printf("Solving Self-Similar Solution\n");
    SelfSimilarSol simfunc;
    init_SelfSimilarSol(&simfunc, self->gamma, self->xi_arr->data[npts-1]);
    for (size_t i = 0; i < npts; ++i) {
        simfunc.xi = self->xi_arr->data[i];
        // perform minimization: force bounds on V to keep the solution stable
        int status = minimize(initial_guess, 
            lower_bound,
            upper_bound,
            1000,
            &simfunc
        );

        double Vtemp = initial_guess;

        // save the solution and calculate residual
        self->sols[0] = simfunc.Z_rhs(&simfunc, Vtemp);
        self->sols[1] = Vtemp;
        self->sols[2] = simfunc.G_rhs(&simfunc, Vtemp);
        residual(&simfunc, self->sols, self->residuals);
    }
    printf("Self-Similar Solution reached for %zu pts", npts);
}

void free_TaylorSol(TaylorSol * self) {
    // free TaylorSol memory
    gsl_block_free(self->xi_arr);
    gsl_block_free(self->Z_arr);
    gsl_block_free(self->G_arr);
    gsl_block_free(self->V_arr);
    free(self);
}

int minimize(double guess, double lower_bound, double upper_bound, size_t max_iter, SelfSimilarSol * self)
{
    // minimize the solution
    int status;
    size_t iter = 0;
    const gsl_min_fminimizer_type *T;
    gsl_min_fminimizer *s;
    double m_expected = M_PI;
    gsl_function F;

    F.function = &obj_function;
    F.params = self;

    T = gsl_min_fminimizer_brent;
    s = gsl_min_fminimizer_alloc(T);
    gsl_min_fminimizer_set(s, &F, guess, lower_bound, upper_bound);

    printf("using %s method\n",
            gsl_min_fminimizer_name(s));

    printf("%5s [%9s, %9s] %9s %10s %9s\n",
            "iter", "lower", "upper", "min",
            "err", "err(est)");

    printf("%5zu [%.7f, %.7f] %.7f %+.7f %.7f\n",
            iter, lower_bound, upper_bound,
            guess, guess - m_expected, upper_bound - lower_bound);

    do {
        iter++;
        status = gsl_min_fminimizer_iterate(s);

        guess = gsl_min_fminimizer_x_minimum(s);
        lower_bound = gsl_min_fminimizer_x_lower(s);
        upper_bound = gsl_min_fminimizer_x_upper(s);

        status = gsl_min_test_interval(lower_bound, upper_bound, 0.001, 0.0);

        if (status == GSL_SUCCESS)
            printf ("Converged:\n");

        printf("%5zu [%.7f, %.7f] "
                "%.7f %+.7f %.7f\n",
                iter, lower_bound, upper_bound,
                guess, guess - m_expected, upper_bound - lower_bound);
    }
    while (status == GSL_CONTINUE && iter < max_iter);

    gsl_min_fminimizer_free(s);
    
    return status;
}