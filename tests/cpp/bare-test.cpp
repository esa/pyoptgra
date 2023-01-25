/*
 * Copyright 2008, 2021 European Space Agency
 *
 * This file is part of pyoptgra, a pygmo affiliated library.
 *
 * This Source Code Form is available under two different licenses.
 * You may choose to license and use it under version 3 of the
 * GNU General Public License or under the
 * ESA Software Community Licence (ESCL) 2.4 Weak Copyleft.
 * We explicitly reserve the right to release future versions of
 * Pyoptgra and Optgra under different licenses.
 * If copies of GPL3 and ESCL 2.4 were not distributed with this
 * file, you can obtain them at https://www.gnu.org/licenses/gpl-3.0.txt
 * and https://essr.esa.int/license/european-space-agency-community-license-v2-4-weak-copyleft
 */

#include <iostream>
#include <vector>
#include <tuple>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "wrapper.hpp"
#include "test-callables.hpp"

using std::cout;
using std::endl;
using std::vector;

using namespace optgra;
using Catch::Approx;

TEST_CASE("Low-level C-Fortran interface works and computes sensitivity matrices", "[bare-interface]")
{
    const std::vector<int> variable_types = {0};
    const std::vector<int> constraint_types = {-1, -1};
    std::vector<double> x = {10};

    fitness_callback fitness = f_simple;
    gradient_callback gradient = g_simple;
    int max_iterations = 150;                          // MAXITE
    int max_correction_iterations = 90;                // CORITE
    double max_distance_per_iteration = 1;             // VARMAX
    double perturbation_for_snd_order_derivatives = 1; // VARSND)

    int num_variables = variable_types.size();
    int num_constraints = constraint_types.size() - 1;
    std::vector<double> autodiff_deltas = std::vector<double>(num_variables, 0.01);
    int derivatives_computation = 2;

    oginit_(&num_variables, &num_constraints);
    ogctyp_(constraint_types.data());
    ogderi_(&derivatives_computation, autodiff_deltas.data());
    ogdist_(&max_distance_per_iteration, &perturbation_for_snd_order_derivatives);

    ogvtyp_(variable_types.data());

    // Haven't figured out what the others do, but maxiter is an upper bound anyway
    int otheriters = max_iterations; // TODO: figure out what it does.
    ogiter_(&max_iterations, &max_correction_iterations, &otheriters, &otheriters, &otheriters);

    int optimization_method = 2;
    ogomet_(&optimization_method);

    int log_unit = 6;
    int log_level = 1;
    ogwlog_(&log_unit, &log_level);

    int pygmo_log_unit = 7;
    int verbosity = 1;
    ogplog_(&pygmo_log_unit, &verbosity);

    int finopt = 0;
    int finite = 0;

    std::vector<double> valvar(x);
    std::vector<double> valcon(num_constraints + 1);

    static_callable_store::set_fitness_callable(fitness);
    static_callable_store::set_gradient_callable(gradient);
    static_callable_store::set_x_dim(num_variables);
    static_callable_store::set_c_dim(num_constraints + 1);

    // ogexec_(valvar.data(), valcon.data(), &finopt, &finite,
    //  static_callable_store::fitness, static_callable_store::gradient);

    int sensitivity_mode = -1;
    ogsopt_(&sensitivity_mode);

    valvar = x;
    ogexec_(valvar.data(), valcon.data(), &finopt, &finite,
            static_callable_store::fitness, static_callable_store::gradient);

    // call and output oggsst to compare
    vector<double> senvar(num_variables);
    vector<double> senqua(num_constraints + 1);
    vector<double> sencon(num_constraints + 1);
    vector<int> senact(num_constraints + 1);
    vector<double> sender((num_constraints + 1) * num_variables);
    vector<int> actcon(num_constraints + 1);
    vector<int> conact(num_constraints + 4);
    vector<double> conred((num_constraints + 3) * num_variables);
    vector<double> conder((num_constraints + 3) * num_variables);
    int numact = 0;

    oggsst_(senvar.data(), senqua.data(), sencon.data(), senact.data(), sender.data(), actcon.data(), conact.data(), conred.data(), conder.data(), &numact);

    cout << "senact: ";
    for (int i = 0; i < num_constraints + 1; i++)
    {
        cout << senact[i] << " ";
    }
    cout << endl;

    cout << "actcon: ";
    for (int i = 0; i < num_constraints + 1; i++)
    {
        cout << senact[i] << " ";
    }
    cout << endl;

    cout << "conact: ";
    for (int i = 0; i < num_constraints + 1; i++)
    {
        cout << senact[i] << " ";
    }
    cout << endl;

    int x_dim = num_variables;
    std::vector<int> constraint_status(num_constraints);
    std::vector<double> concon((num_constraints + 1) * num_constraints);
    std::vector<double> convar((num_constraints + 1) * x_dim);
    std::vector<double> varcon(x_dim * num_constraints);
    std::vector<double> varvar(x_dim * x_dim);

    // call ogsens
    ogsens_(constraint_status.data(), concon.data(), convar.data(), varcon.data(), varvar.data());

    // allocate unflattened sensitivity matrices
    std::vector<std::vector<double>> constraints_to_active_constraints(num_constraints + 1);
    std::vector<std::vector<double>> constraints_to_parameters(num_constraints + 1);
    std::vector<std::vector<double>> variables_to_active_constraints(x_dim);
    std::vector<std::vector<double>> variables_to_parameters(x_dim);

    // copy values for constraints_to_active_constraints and constraints_to_parameters
    for (int i = 0; i < (num_constraints + 1); i++)
    {
        constraints_to_active_constraints[i].resize(num_constraints);
        constraints_to_parameters[i].resize(x_dim);

        for (int j = 0; j < num_constraints; j++)
        {
            constraints_to_active_constraints[i][j] = concon[j * num_constraints + i];
        }

        for (int j = 0; j < x_dim; j++)
        {
            constraints_to_parameters[i][j] = convar[j * num_constraints + i];
        }
    }

    // copy values for variables_to_active_constraints and variables_to_parameters
    for (int i = 0; i < x_dim; i++)
    {
        variables_to_active_constraints[i].resize(num_constraints);
        variables_to_parameters[i].resize(x_dim);

        for (int j = 0; j < num_constraints; j++)
        {
            variables_to_active_constraints[i][j] = varcon[j * x_dim + i];
        }

        for (int j = 0; j < x_dim; j++)
        {
            variables_to_parameters[i][j] = varvar[j * x_dim + i];
        }
    }

    ogclos_();
}

TEST_CASE("Sensitivity update with constraint delta is computed", "[bare-update-delta]")
{
    const std::vector<int> variable_types = {0};
    const std::vector<int> &constraint_types = {-1, -1};

    std::vector<double> x = {10};
    fitness_callback fitness = f_simple;
    gradient_callback gradient = g_simple;
    std::vector<double> delta = {-2};
    int max_iterations = 150;                          // MAXITE
    int max_correction_iterations = 90;                // CORITE
    double max_distance_per_iteration = 10;            // VARMAX
    double perturbation_for_snd_order_derivatives = 1; // VARSND)

    int num_variables = variable_types.size();
    int num_constraints = constraint_types.size() - 1;
    std::vector<double> autodiff_deltas = std::vector<double>(num_variables, 0.01);
    int derivatives_computation = 2;

    oginit_(&num_variables, &num_constraints);
    ogctyp_(constraint_types.data());
    ogderi_(&derivatives_computation, autodiff_deltas.data());
    ogdist_(&max_distance_per_iteration, &perturbation_for_snd_order_derivatives);

    ogvtyp_(variable_types.data());

    // Haven't figured out what the others do, but maxiter is an upper bound anyway
    int otheriters = max_iterations; // TODO: figure out what it does.
    ogiter_(&max_iterations, &max_correction_iterations, &otheriters, &otheriters, &otheriters);

    int optimization_method = 2;
    ogomet_(&optimization_method);

    int log_unit = 6;
    int log_level = 1;
    ogwlog_(&log_unit, &log_level);

    int pygmo_log_unit = 7;
    int verbosity = 1;
    ogplog_(&pygmo_log_unit, &verbosity);

    int finopt = 0;
    int finite = 0;

    std::vector<double> valvar(x);
    std::vector<double> valcon(num_constraints + 1);

    static_callable_store::set_fitness_callable(fitness);
    static_callable_store::set_gradient_callable(gradient);
    static_callable_store::set_x_dim(num_variables);
    static_callable_store::set_c_dim(num_constraints + 1);

    ogexec_(valvar.data(), valcon.data(), &finopt, &finite,
            static_callable_store::fitness, static_callable_store::gradient);

    int sensitivity_mode = -1;
    ogsopt_(&sensitivity_mode);

    valvar = x;
    ogexec_(valvar.data(), valcon.data(), &finopt, &finite,
            static_callable_store::fitness, static_callable_store::gradient);

    sensitivity_mode = 2;
    ogsopt_(&sensitivity_mode);

    ogcdel_(delta.data());

    // valvar = x;
    ogexec_(valvar.data(), valcon.data(), &finopt, &finite,
            static_callable_store::fitness, static_callable_store::gradient);

    ogclos_();
    // check that x was moved
    REQUIRE(valvar[0] == Approx(12.0));
}