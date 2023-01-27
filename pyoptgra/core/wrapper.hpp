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

#pragma once

#include <stdexcept>
#include <vector>
#include <tuple>
#include <iostream>
#include <functional>
#include <numeric>
#include <mutex>

extern "C"
{
    void ogcdel_(double *delcon);
    void ogclos_();
    void ogcpri_(int *pricon);
    void ogcsca_(double *scacon);
    void ogcstr_(char **strcon, int *lencon);
    void ogctyp_(const int *contyp);
    void ogderi_(int *dervar, double *pervar);
    void ogdist_(double *maxvar, double *sndvar);
    void ogeval_(double *valvar, double *valcon, int *dervar, double *dercon,
                 void (*)(double *, double *), void (*)(double *, double *, double *));
    void ogexec_(double *valvar, double *valcon, int *finopt, int *finite,
                 void (*)(double *, double *, int *), void (*)(double *, double *, double *));
    void oggsst_(double *senvar, double *senqua, double *sencon, int *senact, double *sender,
                 int *actcon, int *conact, double *conred, double *conder, int *actnum);
    void oginit_(int *varnum, int *connum);
    void ogiter_(int *itemax, int *itecor, int *iteopt, int *itediv, int *itecnv);
    void ogomet_(int *metopt);
    void ogsens_(int *consta, double *concon, double *convar, double *varcon, double *varvar);
    void ogsopt_(int *optsen);
    void ogssst_(const double *senvar, const double *senqua, const double *sencon, const int *senact, const double *sender,
                 const int *actcon, const int *conact, const double *conred, double *conder, int *actnum);
    void ogvsca_(double *scavar);
    void ogvtyp_(const int *vartyp);
    void ogvstr_(char **strvar, int *lenvar);
    void ogwlog_(int *lunlog, int *levlog);
    void ogplog_(int *luplog, int *bosver);
}

namespace optgra
{

    typedef std::function<std::vector<double>(std::vector<double>)> fitness_callback;

    typedef std::function<std::vector<std::vector<double>>(std::vector<double>)> gradient_callback;

    using std::function;
    using std::tuple;
    using std::vector;

    // senvar, senqua, sencon, senact, sender, actcon, conact, conred, conder
    typedef tuple<vector<double>, vector<double>, vector<double>, vector<int>, vector<double>, vector<int>, vector<int>, vector<double>, vector<double>> sensitivity_state;

    /** This struct is just to connect the std::functions passed from python
     *  to the unholy mess of static function pointers which are required by Fortran.
     *  It is emphatically not thread safe.
     */
    struct static_callable_store
    {

        static void fitness(double *x, double *out_f, int *inapplicable_flag)
        {
            std::vector<double> x_vector(x_dim);
            std::copy(x, x + x_dim, x_vector.begin());

            std::vector<double> fitness_vector;
            try
            {
                fitness_vector = f_callable(x_vector);
            }
            catch (const std::bad_function_call &e)
            {
                throw(std::runtime_error("Empty fitness function was called."));
            }

            if (int(fitness_vector.size()) != c_dim)
            {
                throw(std::invalid_argument("Got vector of size" + std::to_string(fitness_vector.size()) + " from fitness callable, but expected " + std::to_string(c_dim) + " constraints+fitness."));
            }

            std::copy(fitness_vector.begin(), fitness_vector.end(), out_f);
        }

        static void gradient(double *x, double *out_f, double *out_derivatives)
        {
            fitness(x, out_f, 0); // this can probably be optimized

            std::vector<double> x_vector(x_dim);
            std::copy(x, x + x_dim, x_vector.begin());
            std::vector<std::vector<double>> gradient_vector;
            try
            {
                gradient_vector = g_callable(x_vector);
            }
            catch (const std::bad_function_call &e)
            {
                throw(std::runtime_error("Empty gradient function was called."));
            }

            int num_constraints = gradient_vector.size();
            if (num_constraints != c_dim)
            {
                throw(std::invalid_argument("Got vector of size" + std::to_string(num_constraints) + " from gradient callable, but expected " + std::to_string(c_dim) + " constraints+fitness."));
            }

            for (int i = 0; i < num_constraints; i++)
            {
                if (int(gradient_vector[i].size()) != x_dim)
                {
                    throw(std::invalid_argument("Got vector of size" + std::to_string(int(gradient_vector[i].size())) + " from row " + std::to_string(i) + " of gradient callable, but expected " + std::to_string(x_dim) + " variables."));
                }
                for (int j = 0; j < x_dim; j++)
                {
                    out_derivatives[j * num_constraints + i] = gradient_vector[i][j];
                }
            }
        }

        static void set_fitness_callable(fitness_callback f)
        {
            f_callable = f;
        }

        static void set_gradient_callable(gradient_callback g)
        {
            g_callable = g;
        }

        static void set_x_dim(int dim)
        {
            x_dim = dim;
        }

        static void set_c_dim(int dim)
        {
            c_dim = dim;
        }

        static fitness_callback f_callable;
        static gradient_callback g_callable;
        static int x_dim;
        static int c_dim;
    };
    // static initialization
    fitness_callback static_callable_store::f_callable;
    gradient_callback static_callable_store::g_callable;
    int static_callable_store::x_dim;
    int static_callable_store::c_dim;

    struct optgra_raii
    {

        optgra_raii() = delete;

        optgra_raii(const optgra_raii &) = delete;

        optgra_raii(const std::vector<int> &variable_types, const std::vector<int> &constraint_types,
                    int max_iterations = 150,                          // MAXITE
                    int max_correction_iterations = 90,                // CORITE
                    double max_distance_per_iteration = 10,            // VARMAX
                    double perturbation_for_snd_order_derivatives = 1, // VARSND
                    std::vector<double> convergence_thresholds = {},
                    std::vector<double> variable_scaling_factors = {},
                    std::vector<int> constraint_priorities = {},
                    std::vector<std::string> variable_names = {},
                    std::vector<std::string> constraint_names = {},
                    int optimization_method = 2,     // OPTMET
                    int derivatives_computation = 1, // VARDER
                    std::vector<double> autodiff_deltas = {},
                    int log_level = 1,
                    int verbosity = 0)
        {
            num_variables = variable_types.size();

            num_constraints = constraint_types.size() - 1;
            if (autodiff_deltas.size() == 0)
            {
                autodiff_deltas = std::vector<double>(num_variables, 0.01);
            }
            else if (autodiff_deltas.size() != num_variables)
            {
                throw(std::invalid_argument("Got " + std::to_string(autodiff_deltas.size()) + " autodiff deltas for " + std::to_string(num_variables) + " variables."));
            }

            // TODO: more sanity checks for parameters.

            // Ensure that at most one optgra_raii object is active at the same time
            optgra_mutex.lock();

            oginit_(&num_variables, &num_constraints);
            ogctyp_(constraint_types.data());
            ogderi_(&derivatives_computation, autodiff_deltas.data());
            ogdist_(&max_distance_per_iteration, &perturbation_for_snd_order_derivatives);

            ogvtyp_(variable_types.data());

            // Haven't figured out what the others do, but maxiter is an upper bound anyway
            int otheriters = max_iterations; // TODO: figure out what it does.
            ogiter_(&max_iterations, &max_correction_iterations, &otheriters, &otheriters, &otheriters);

            ogomet_(&optimization_method);

            // original OPTGRA screen output configuration
            int log_unit = 6;
            ogwlog_(&log_unit, &log_level);

            // pygmo style screen output configuration
            ogplog_(&log_unit, &verbosity);

            if (variable_scaling_factors.size() > 0)
            {
                if (variable_scaling_factors.size() != num_variables)
                {
                    throw(std::invalid_argument("Got " + std::to_string(variable_scaling_factors.size()) + " scaling factors for " + std::to_string(num_variables) + " variables."));
                }

                ogvsca_(variable_scaling_factors.data());
            }

            if (convergence_thresholds.size() > 0)
            {
                if (convergence_thresholds.size() != constraint_types.size())
                {
                    throw(std::invalid_argument("Got " + std::to_string(convergence_thresholds.size()) + " convergence thresholds for " + std::to_string(constraint_types.size()) + " constraints+fitness."));
                }
                if (!convergence_thresholds[convergence_thresholds.size() - 1] > 0)
                {
                    throw(std::invalid_argument("Convergence threshold for merit function must be positive."));
                }
                ogcsca_(convergence_thresholds.data());
            }

            if (constraint_priorities.size() > 0)
            {
                if (constraint_priorities.size() != constraint_types.size())
                {
                    // TODO: Find out what the last priority is for!
                    throw(std::invalid_argument("Got " + std::to_string(constraint_priorities.size()) + " constraint priorities for " + std::to_string(constraint_types.size()) + " constraints+fitness."));
                }
                ogcpri_(constraint_priorities.data());
            }

            // TODO: figure out how string arrays are passed to fortran for variable names

            initialized_sensitivity = false;
        }

        std::tuple<std::vector<double>, std::vector<double>, int> exec(std::vector<double> initial_x, fitness_callback fitness, gradient_callback gradient)
        {

            if (int(initial_x.size()) != num_variables)
            {
                throw(std::invalid_argument("Expected " + std::to_string(num_variables) + ", but got " + std::to_string(initial_x.size())));
            }

            std::vector<double> valvar(initial_x);
            std::vector<double> valcon(num_constraints + 1);

            static_callable_store::set_fitness_callable(fitness);
            static_callable_store::set_gradient_callable(gradient);
            static_callable_store::set_x_dim(num_variables);
            static_callable_store::set_c_dim(num_constraints + 1);

            int sensitivity_mode = 0;
            ogsopt_(&sensitivity_mode);

            int finopt = 0;
            int finite = 0;
            ogexec_(valvar.data(), valcon.data(), &finopt, &finite,
                    static_callable_store::fitness, static_callable_store::gradient);

            // resetting callables to make sure that passed handles go out of scope
            static_callable_store::set_fitness_callable(fitness_callback());
            static_callable_store::set_gradient_callable(gradient_callback());

            return std::make_tuple(valvar, valcon, finopt);
        }

        std::tuple<std::vector<double>, std::vector<double>, int, int> initialize_sensitivity_data(std::vector<double> x, fitness_callback fitness, gradient_callback gradient)
        {
            if (int(x.size()) != num_variables)
            {
                throw(std::invalid_argument("Expected " + std::to_string(num_variables) + ", but got " + std::to_string(x.size())));
            }

            std::vector<double> valvar(x);
            std::vector<double> valcon(num_constraints + 1);

            static_callable_store::set_fitness_callable(fitness);
            static_callable_store::set_gradient_callable(gradient);
            static_callable_store::set_x_dim(num_variables);
            static_callable_store::set_c_dim(num_constraints + 1);

            int sensitivity_mode = -1;
            ogsopt_(&sensitivity_mode);

            int finopt = 0;
            int finite = 0;
            ogexec_(valvar.data(), valcon.data(), &finopt, &finite,
                    static_callable_store::fitness, static_callable_store::gradient);

            initialized_sensitivity = true; // TODO: check return values before setting it to true

            // resetting callables to make sure that passed handles go out of scope
            static_callable_store::set_fitness_callable(fitness_callback());
            static_callable_store::set_gradient_callable(gradient_callback());

            return std::make_tuple(valvar, valcon, finopt, finite);
        }

        std::tuple<std::vector<int>, std::vector<std::vector<double>>, std::vector<std::vector<double>>,
                   std::vector<std::vector<double>>, std::vector<std::vector<double>>>
        sensitivity_matrices()
        {

            if (!initialized_sensitivity)
            {
                throw(std::runtime_error("Please call initialize_sensitivity_data first."));
            }

            // allocate flattened sensitivity matrices
            int x_dim = num_variables;
            std::vector<int> constraint_status(num_constraints);
            std::vector<double> concon((num_constraints + 1) * num_constraints);
            std::vector<double> convar((num_constraints + 1) * x_dim);
            std::vector<double> varcon(x_dim * num_constraints);
            std::vector<double> varvar(x_dim * x_dim);

            // call ogsens
            ogsens_(constraint_status.data(), concon.data(), convar.data(), varcon.data(), varvar.data());
            /**
            C OUT | CONSTA(NUMCON)   | I*4 | CONSTRAINT STATUS (0=PAS 1=ACT)
            C OUT | CONCON(NUMCON+1, | R*8 | SENSITIVITY OF CONTRAINTS+MERIT W.R.T.
            C     |        NUMCON)   |     |                ACTIVE CONSTRAINTS
            C OUT | CONVAR(NUMCON+1, | R*8 | SENSITIVITY OF CONTRAINTS+MERIT W.R.T.
            C     |        NUMVAR)   |     |                PARAMETERS
            C OUT | VARCON(NUMVAR  , | R*8 | SENSITIVITY OF VARIABLES W.R.T.
            C     |        NUMCON)   |     |                ACTIVE CONSTRAINTS
            C OUT | VARVAR(NUMVAR  , | R*8 | SENSITIVITY OF VARIABLES W.R.T.
            C     |        NUMVAR)   |     |                PARAMETERS
            */

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

            return std::make_tuple(constraint_status, constraints_to_active_constraints, constraints_to_parameters,
                                   variables_to_active_constraints, variables_to_parameters);
        }

        std::tuple<std::vector<double>, std::vector<double>, int> sensitivity_update(fitness_callback fitness, gradient_callback gradient)
        {

            if (!initialized_sensitivity)
            {
                throw(std::runtime_error("Please call initialize_sensitivity_data first."));
            }

            std::vector<double> valvar(num_variables);
            std::vector<double> valcon(num_constraints + 1);

            static_callable_store::set_fitness_callable(fitness);
            static_callable_store::set_gradient_callable(gradient);
            static_callable_store::set_x_dim(num_variables);
            static_callable_store::set_c_dim(num_constraints + 1);

            int sensitivity_mode = 1;
            ogsopt_(&sensitivity_mode);

            int finopt = 0;
            int finite = 0;
            ogexec_(valvar.data(), valcon.data(), &finopt, &finite,
                    static_callable_store::fitness, static_callable_store::gradient);

            // resetting callables to make sure that passed handles go out of scope
            static_callable_store::set_fitness_callable(fitness_callback());
            static_callable_store::set_gradient_callable(gradient_callback());

            return std::make_tuple(valvar, valcon, finopt);
        }

        std::tuple<std::vector<double>, std::vector<double>, int> sensitivity_update_delta(std::vector<double> constraint_delta)
        {

            if (!initialized_sensitivity)
            {
                throw(std::runtime_error("Please call initialize_sensitivity_data first."));
            }

            if (int(constraint_delta.size()) != num_constraints)
            {
                throw(std::invalid_argument("Expected " + std::to_string(num_constraints) + " constraint deltas, but got " + std::to_string(constraint_delta.size())));
            }

            std::vector<double> valvar(num_variables);
            std::vector<double> valcon(num_constraints + 1);

            int sensitivity_mode = 2;
            ogsopt_(&sensitivity_mode);

            ogcdel_(constraint_delta.data());

            int finopt = 0;
            int finite = 0;
            ogexec_(valvar.data(), valcon.data(), &finopt, &finite,
                    static_callable_store::fitness, static_callable_store::gradient);

            return std::make_tuple(valvar, valcon, finopt);
        }

        void set_sensitivity_state_data(sensitivity_state state_tuple)
        {

            vector<double> senvar;
            vector<double> senqua;
            vector<double> sencon;
            vector<int> senact;
            vector<double> sender;
            vector<int> actcon;
            vector<int> conact;
            vector<double> conred;
            vector<double> conder;

            std::tie(senvar, senqua, sencon, senact, sender, actcon, conact, conred, conder) = state_tuple;

            if (int(senvar.size()) != num_variables)
            {
                throw(std::invalid_argument("First vector needs to be of size num_variables."));
            }

            if (int(senqua.size()) != num_constraints + 1)
            {
                throw(std::invalid_argument("Second, third and fourth vector need to be of size num_constraints+1."));
            }

            if (int(sencon.size()) != num_constraints + 1)
            {
                throw(std::invalid_argument("Second, third and fourth vector need to be of size num_constraints+1."));
            }

            if (int(senact.size()) != num_constraints + 1)
            {
                throw(std::invalid_argument("Second, third and fourth vector need to be of size num_constraints+1."));
            }

            if (int(sender.size()) != ((num_constraints + 1) * num_variables))
            {
                throw(std::invalid_argument("Fifth vector needs to be of size (num_constraints+1)*num_variables."));
            }

            int numact = std::accumulate(senact.begin(), senact.end(), 0);

            // TODO: check sizes of actcon, conact, conred

            ogssst_(senvar.data(), senqua.data(), sencon.data(), senact.data(), sender.data(), actcon.data(), conact.data(), conred.data(), conder.data(), &numact);

            initialized_sensitivity = true;
        }

        sensitivity_state get_sensitivity_state_data() const
        {
            if (!initialized_sensitivity)
            {
                throw(std::runtime_error("Please call initialize_sensitivity_data first."));
            }

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
            int measured_numact = std::accumulate(senact.begin(), senact.end(), 0);

            if (numact != measured_numact)
            {
                std::cout << "Warning: Got " << measured_numact << " constraints reported as active, but numact is " << numact << std::endl;
            }

            return std::make_tuple(senvar, senqua, sencon, senact, sender, actcon, conact, conred, conder);
        }

        ~optgra_raii()
        {
            ogclos_();
            optgra_mutex.unlock();
        }

    private:
        int num_variables;
        int num_constraints;
        bool initialized_sensitivity;

        static std::mutex optgra_mutex;
    };

    std::mutex optgra_raii::optgra_mutex;

    /// Main C++ wrapper function
    /**
     * Call optgra to optimize a problem. Most of the parameters are identical to the constructor arguments of pyoptgra,
     *    but some additional ones are available.
     *
     * @param initial_x the initial guess for the decision vector
     * @param constraint_types types of constraints. Set 0 for equality constraints,
     *    -1 for inequality constraints that should be negative, 1 for positive inequality constraints and -2 for unenforced constraints.
     *    Last element describes the merit function, with -1 for minimization problems and 1 for maximization problems.
     * @param fitness a callable for the fitness values. It is called with the current x,
     *    expected output is an array of first all equality constraints, then all inequality constraints, and last the merit function
     * @param gradient a callable for the gradient values, optional. It is called with the current x,
     *    expected output is a two-dimensional array g, with g_ij being the gradient of constraint i with respect to input variable j.
     * @param has_gradient whether the problem has a gradient. If set to False, the gradient callable will not be called
     *    and numerical differentiation will be used instead.
     * @param max_iterations the maximum number of iterations. Optional, defaults to 150.
     * @param max_correction_iterations number of constraint correction iterations in the beginning.
     *    If no feasible solution is found within that many iterations, Optgra aborts. Optional, defaults to 90.
     * @param max_distance_per_iteration maximum scaled distance traveled in each iteration. Optional, defaults to 10
     * @param perturbation_for_snd_order_derivatives used as delta for numerically computing second order errors
     *    of the constraints in the optimization step. Parameter VARSND in Fortran. Optional, defaults to 1
     * @param convergence_thresholds tolerance a constraint can deviate and still be considered fulfilled.
     *    Constraints with lower thresholds will be prioritized during optimization. Thresholds of 0 break the optimization process.
     * @param variable_scaling_factors scaling factors for the input variables.
     *    If passed, must be positive and as many as there are variables
     * @param constraint_priorities filter in which to consider constraints. Lower constraint priorities are fulfilled earlier.
     *    During the initial constraint correction phase, only constraints with a priority at most k
     *    are considered in iteration k. Defaults to zero, so that all constraints are considered
     *    from the beginning.
     * @param variable_names Not yet implemented
     * @param constraint_names Not yet implemented
     * @param optimization_method select 0 for steepest descent, 1 for modified spectral conjugate gradient method,
     *    2 for spectral conjugate gradient method and 3 for conjugate gradient method. Parameter OPTMET in Fortran.
     * @param derivatives_computation method to compute gradients. 0 is no gradient, 1 is the user-defined gradient function,
     *    2 is a numerical gradient with double differencing, 3 a numerical gradient with single differencing.
     *    Parameter VARDER in Fortran.
     * @param autodiff_deltas deltas used for each variable when computing the gradient numerically. Optional, defaults to 0.001.
     * @param variable_types Optional array, specifiying 0 (normal, default) or 1 (fixed, only used for sensitivity) for each variable.
     * @param log_level original OPTGRA logging output: 0 has no output, 4 and higher have maximum output. Set this to 0 if you want to use the pygmo
          logging system based on `set_verbosity()`.
     * @param verbosity pygmo-style logging output: 0 has no output, N means an output every Nth cost function evaluation. Set `log_level` to zero to use this.
     *
     * @return a tuple of the best value of x, the fitness of that x, and a status flag of optgra
     *
     * @throws unspecified any exception thrown by memory errors in standard containers
     */
    std::tuple<std::vector<double>, std::vector<double>, int> optimize(const std::vector<double> &initial_x,
                                                                       const std::vector<int> &constraint_types, fitness_callback fitness, gradient_callback gradient, bool has_gradient,
                                                                       int max_iterations = 150,                          // MAXITE
                                                                       int max_correction_iterations = 90,                // CORITE
                                                                       double max_distance_per_iteration = 10,            // VARMAX
                                                                       double perturbation_for_snd_order_derivatives = 1, // VARSND
                                                                       std::vector<double> convergence_thresholds = {},
                                                                       std::vector<double> variable_scaling_factors = {},
                                                                       std::vector<int> constraint_priorities = {},
                                                                       std::vector<std::string> variable_names = {},
                                                                       std::vector<std::string> constraint_names = {},
                                                                       int optimization_method = 2,     // OPTMET
                                                                       int derivatives_computation = 1, // VARDER
                                                                       std::vector<double> autodiff_deltas = {},
                                                                       std::vector<int> variable_types = {},
                                                                       int log_level = 1,
                                                                       int verbosity = 0)
    {
        // initialization
        int num_variables = initial_x.size();

        if (variable_types.size() == 0)
        {
            variable_types = std::vector<int>(num_variables, 0);
        }

        if (variable_types.size() != initial_x.size())
        {
            throw(std::invalid_argument("Got initial_x vector of size" + std::to_string(initial_x.size()) + " but variable_types vector of size " + std::to_string(variable_types.size()) + "."));
        }

        if (derivatives_computation == 1 && !has_gradient)
        {
            std::cout << "No user-defined gradient available, switching to numeric differentiation." << std::endl;
            derivatives_computation = 3;
        }

        optgra_raii raii_object(variable_types, constraint_types,
                                max_iterations,                         // MAXITE
                                max_correction_iterations,              // CORITE
                                max_distance_per_iteration,             // VARMAX
                                perturbation_for_snd_order_derivatives, // VARSND
                                convergence_thresholds,
                                variable_scaling_factors,
                                constraint_priorities,
                                variable_names,
                                constraint_names,
                                optimization_method,     // OPTMET
                                derivatives_computation, // VARDER
                                autodiff_deltas,
                                log_level,
                                verbosity);

        return raii_object.exec(initial_x, fitness, gradient);
    }

    /// Compute sensitivity state and matrices in one go, without creating a sensitivity state tuple.
    /**
    * @param x the decision vector around which the sensitivity analysis is to be performed
    * @param constraint_types types of constraints. Set 0 for equality constraints,
    *    -1 for inequality constraints that should be negative, 1 for positive inequality constraints and -2 for unenforced constraints
    *    Last element describes the merit function, with -1 for minimization problems and 1 for maximization problems.
    * @param fitness a callable for the fitness values. It is called with the current x,
    *    expected output is an array of first all equality constraints, then all inequality constraints, and last the merit function
    * @param gradient a callable for the gradient values, optional. It is called with the current x,
    *    expected output is a two-dimensional array g, with g_ij being the gradient of constraint i with respect to input variable j.
    * @param has_gradient whether the problem has a gradient. If set to False, the gradient callable will not be called
    *    and numerical differentiation will be used instead.
    * @param max_distance_per_iteration maximum scaled distance traveled in each iteration. Optional, defaults to 10
    * @param perturbation_for_snd_order_derivatives used as delta for numerically computing second order errors
    *    of the constraints in the optimization step. Parameter VARSND in Fortran. Optional, defaults to 1
    * @param variable_scaling_factors scaling factors for the input variables.
    *    If passed, must be positive and as many as there are variables
    * @param variable_names Not yet implemented
    * @param constraint_names Not yet implemented
    * @param derivatives_computation method to compute gradients. 0 is no gradient, 1 is the user-defined gradient function,
    *    2 is a numerical gradient with double differencing, 3 a numerical gradient with single differencing.
    *    Parameter VARDER in Fortran.
    * @param autodiff_deltas deltas used for each variable when computing the gradient numerically. Optional, defaults to 0.001.
    * @param variable_types Optional array, specifiying 0 (normal, default) or 1 (fixed, only used for sensitivity) for each variable.
    * @param log_level original OPTGRA logging output: 0 has no output, 4 and higher have maximum output
    * @param verbosity pygmo-style logging output: 0 has no output, N means an output every Nth cost function evaluation
    *
    * @return  A tuple of one list and four matrices: a boolean list of whether each constraint is active,
                the sensitivity of constraints + merit function with respect to active constraints,
                the sensitivity of constraints + merit function with respect to parameters,
                the sensitivity of variables with respect to active constraints,
                and the sensitivity of variables with respect to parameters.
    */
    std::tuple<std::vector<int>, std::vector<std::vector<double>>, std::vector<std::vector<double>>,
               std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    compute_sensitivity_matrices(const std::vector<double> &x,
                                 const std::vector<int> &constraint_types, fitness_callback fitness, gradient_callback gradient, bool has_gradient,
                                 double max_distance_per_iteration = 10,            // VARMAX
                                 double perturbation_for_snd_order_derivatives = 1, // VARSND
                                 std::vector<double> variable_scaling_factors = {},
                                 std::vector<std::string> variable_names = {},
                                 std::vector<std::string> constraint_names = {},
                                 int derivatives_computation = 1, // VARDER
                                 std::vector<double> autodiff_deltas = {},
                                 std::vector<int> variable_types = {},
                                 int log_level = 1,
                                 int verbosity = 0)
    {

        int num_variables = x.size();

        if (variable_types.size() == 0)
        {
            variable_types = std::vector<int>(num_variables, 0);
        }

        if (variable_types.size() != x.size())
        {
            throw(std::invalid_argument("Got initial_x vector of size" + std::to_string(x.size()) + " but variable_types vector of size " + std::to_string(variable_types.size()) + "."));
        }

        if (derivatives_computation == 1 && !has_gradient)
        {
            std::cout << "No user-defined gradient available, switching to numeric differentiation." << std::endl;
            derivatives_computation = 3;
        }

        optgra_raii raii_object(variable_types, constraint_types,
                                1,                                      // max_iterations, // MAXITE
                                1,                                      // max_correction_iterations, // CORITE
                                max_distance_per_iteration,             // VARMAX
                                perturbation_for_snd_order_derivatives, // VARSND
                                {},                                     // convergence_thresholds,
                                variable_scaling_factors,
                                {}, // constraint_priorities,
                                variable_names,
                                constraint_names,
                                2,                       // optimization_method, // OPTMET
                                derivatives_computation, // VARDER
                                autodiff_deltas,
                                log_level,
                                verbosity);

        raii_object.initialize_sensitivity_data(x, fitness, gradient);

        return raii_object.sensitivity_matrices();
    }

    /// Create a state tuple usable for sensitivity updates
    /**
     * @param x the decision vector around which the sensitivity analysis is to be performed
     * @param constraint_types types of constraints. Set 0 for equality constraints,
     *    -1 for inequality constraints that should be negative, 1 for positive inequality constraints and -2 for unenforced constraints
     *    Last element describes the merit function, with -1 for minimization problems and 1 for maximization problems.
     * @param fitness a callable for the fitness values. It is called with the current x,
     *    expected output is an array of first all equality constraints, then all inequality constraints, and last the merit function
     * @param gradient a callable for the gradient values, optional. It is called with the current x,
     *    expected output is a two-dimensional array g, with g_ij being the gradient of constraint i with respect to input variable j.
     * @param has_gradient whether the problem has a gradient. If set to False, the gradient callable will not be called
     *    and numerical differentiation will be used instead.
     * @param max_distance_per_iteration maximum scaled distance traveled in each iteration. Optional, defaults to 10
     * @param perturbation_for_snd_order_derivatives used as delta for numerically computing second order errors
     *    of the constraints in the optimization step. Parameter VARSND in Fortran. Optional, defaults to 1
     * @param variable_scaling_factors scaling factors for the input variables.
     *    If passed, must be positive and as many as there are variables
     * @param derivatives_computation method to compute gradients. 0 is no gradient, 1 is the user-defined gradient function,
     *    2 is a numerical gradient with double differencing, 3 a numerical gradient with single differencing.
     *    Parameter VARDER in Fortran.
     * @param autodiff_deltas deltas used for each variable when computing the gradient numerically. Optional, defaults to 0.001.
     * @param variable_types Optional array, specifiying 0 (normal, default) or 1 (fixed, only used for sensitivity) for each variable.
     * @param log_level original OPTGRA logging output: 0 has no output, 4 and higher have maximum output
     * @param verbosity pygmo-style logging output: 0 has no output, N means an output every Nth cost function evaluation
     *
     * @return A tuple of the current sensitivity state and the x for which the sensitivity analysis was performed.
     *            It may be different from the x given as argument, if optimization steps were performed in the meantime.
     */
    std::tuple<sensitivity_state, std::vector<double>> prepare_sensitivity_state(const std::vector<double> &x,
                                                                                 const std::vector<int> &constraint_types, fitness_callback fitness, gradient_callback gradient, bool has_gradient,
                                                                                 double max_distance_per_iteration = 10,            // VARMAX
                                                                                 double perturbation_for_snd_order_derivatives = 1, // VARSND
                                                                                 std::vector<double> variable_scaling_factors = {},
                                                                                 int derivatives_computation = 1, // VARDER
                                                                                 std::vector<double> autodiff_deltas = {},
                                                                                 std::vector<int> variable_types = {},
                                                                                 int log_level = 1,
                                                                                 int verbosity = 0)
    {

        int num_variables = x.size();

        if (variable_types.size() == 0)
        {
            variable_types = std::vector<int>(num_variables, 0);
        }

        if (variable_types.size() != x.size())
        {
            throw(std::invalid_argument("Got initial_x vector of size" + std::to_string(x.size()) + " but variable_types vector of size " + std::to_string(variable_types.size()) + "."));
        }

        if (derivatives_computation == 1 && !has_gradient)
        {
            std::cout << "No user-defined gradient available, switching to numeric differentiation." << std::endl;
            derivatives_computation = 3;
        }

        optgra_raii raii_object(variable_types, constraint_types,
                                1,                                      // max_iterations, // MAXITE
                                1,                                      // max_correction_iterations, // CORITE
                                max_distance_per_iteration,             // VARMAX
                                perturbation_for_snd_order_derivatives, // VARSND
                                {},                                     // convergence_thresholds,
                                variable_scaling_factors,
                                {},                      // constraint_priorities,
                                {},                      // variable_names,
                                {},                      // constraint_names,
                                2,                       // optimization_method, // OPTMET
                                derivatives_computation, // VARDER
                                autodiff_deltas,
                                log_level,
                                verbosity);

        std::vector<double> x_new;
        std::vector<double> y_new;
        std::tie(x_new, y_new, std::ignore, std::ignore) = raii_object.initialize_sensitivity_data(x, fitness, gradient);
        sensitivity_state state = raii_object.get_sensitivity_state_data();

        return std::make_tuple(state, x_new);
    }

    /// Compute sensitivity matrics from a sensitivity state tuple
    /*
    * @param state_tuple A tuple of vectors representing the internal state of optgra prepared for the sensitivity analysis
    * @param variable_types Optional array, specifiying 0 (normal, default) or 1 (fixed, only used for sensitivity) for each variable.
    * @param constraint_types types of constraints. Set 0 for equality constraints,
    *    -1 for inequality constraints that should be negative, 1 for positive inequality constraints and -2 for unenforced constraints
    *
    * @return  A tuple of one list and four matrices: a boolean list of whether each constraint is active,
                the sensitivity of constraints + merit function with respect to active constraints,
                the sensitivity of constraints + merit function with respect to parameters,
                the sensitivity of variables with respect to active constraints,
                and the sensitivity of variables with respect to parameters.
    */
    std::tuple<std::vector<int>, std::vector<std::vector<double>>, std::vector<std::vector<double>>,
               std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    get_sensitivity_matrices(sensitivity_state state_tuple, const std::vector<int> &variable_types,
                             vector<int> constraint_types, double max_distance_per_iteration = 10)
    {

        optgra_raii raii_object(variable_types, constraint_types, 1, 1, max_distance_per_iteration);
        raii_object.set_sensitivity_state_data(state_tuple);
        return raii_object.sensitivity_matrices();
    }

    /// Perform one optimization step with a new fitness callable, starting from the value of x that was set with prepare_sensitivity_state
    /*
     * @param state_tuple A tuple of vectors representing the internal state of optgra prepared for the sensitivity analysis
     * @param variable_types Optional array, specifiying 0 (normal, default) or 1 (fixed, only used for sensitivity) for each variable.
     * @param constraint_types types of constraints. Set 0 for equality constraints,
     *    -1 for inequality constraints that should be negative, 1 for positive inequality constraints and -2 for unenforced constraints
     * @param fitness a callable for the fitness values. It is called with the current x,
     *    expected output is an array of first all equality constraints, then all inequality constraints, and last the merit function
     * @param gradient a callable for the gradient values, optional. It is called with the current x,
     *    expected output is a two-dimensional array g, with g_ij being the gradient of constraint i with respect to input variable j.
     * @param has_gradient whether the problem has a gradient. If set to False, the gradient callable will not be called
     *    and numerical differentiation will be used instead.
     * @param max_distance_per_iteration maximum scaled distance traveled in each iteration. Optional, defaults to 10
     * @param perturbation_for_snd_order_derivatives used as delta for numerically computing second order errors
     *    of the constraints in the optimization step. Parameter VARSND in Fortran. Optional, defaults to 1
     * @param variable_scaling_factors scaling factors for the input variables.
     *    If passed, must be positive and as many as there are variables
     * @param derivatives_computation method to compute gradients. 0 is no gradient, 1 is the user-defined gradient function,
     *    2 is a numerical gradient with double differencing, 3 a numerical gradient with single differencing.
     *    Parameter VARDER in Fortran.
     * @param autodiff_deltas deltas used for each variable when computing the gradient numerically. Optional, defaults to 0.001.
     * @param log_level original OPTGRA logging output: 0 has no output, 4 and higher have maximum output
     * @param verbosity pygmo-style logging output: 0 has no output, N means an output every Nth cost function evaluation
     *
     * @return a tuple of the new value of x, the fitness of that x, and a status flag of optgra
     */
    std::tuple<std::vector<double>, std::vector<double>, int> sensitivity_update_new_callable(sensitivity_state state_tuple, const std::vector<int> &variable_types,
                                                                                              const std::vector<int> &constraint_types, fitness_callback fitness, gradient_callback gradient, bool has_gradient,
                                                                                              double max_distance_per_iteration = 10,            // VARMAX
                                                                                              double perturbation_for_snd_order_derivatives = 1, // VARSND
                                                                                              std::vector<double> variable_scaling_factors = {},
                                                                                              int derivatives_computation = 1, // VARDER
                                                                                              std::vector<double> autodiff_deltas = {},
                                                                                              int log_level = 1,
                                                                                              int verbosity = 0)
    {

        if (derivatives_computation == 1 && !has_gradient)
        {
            std::cout << "No user-defined gradient available, switching to numeric differentiation." << std::endl;
            derivatives_computation = 3;
        }

        // TODO: check consistency of sizes of variable types and variable scaling factors

        optgra_raii raii_object(variable_types, constraint_types,
                                1,                                      // max_iterations, // MAXITE
                                1,                                      // max_correction_iterations, // CORITE
                                max_distance_per_iteration,             // VARMAX
                                perturbation_for_snd_order_derivatives, // VARSND
                                {},                                     // convergence_thresholds,
                                variable_scaling_factors,
                                {},                      // constraint_priorities,
                                {},                      // variable_names,
                                {},                      // constraint_names,
                                2,                       // optimization_method, // OPTMET
                                derivatives_computation, // VARDER
                                autodiff_deltas,
                                log_level,
                                verbosity);

        raii_object.set_sensitivity_state_data(state_tuple);

        return raii_object.sensitivity_update(fitness, gradient);
    }

    /// Perform an update step based on the prepared sensitivity state, without any calls to the fitness callbacks
    /*
     * @param state_tuple A tuple of vectors representing the internal state of optgra prepared for the sensitivity analysis
     * @param variable_types Optional array, specifiying 0 (normal, default) or 1 (fixed, only used for sensitivity) for each variable.
     * @param constraint_types types of constraints. Set 0 for equality constraints,
     *    -1 for inequality constraints that should be negative, 1 for positive inequality constraints and -2 for unenforced constraints
     * @param delta constraint delta which is subtracted from all values of constraints
     * @param max_distance_per_iteration maximum scaled distance traveled in each iteration. Optional, defaults to 10
     * @param perturbation_for_snd_order_derivatives used as delta for numerically computing second order errors
     *    of the constraints in the optimization step. Parameter VARSND in Fortran. Optional, defaults to 1
     * @param variable_scaling_factors scaling factors for the input variables.
     *    If passed, must be positive and as many as there are variables
     * @param log_level original OPTGRA logging output: 0 has no output, 4 and higher have maximum output
     * @param verbosity pygmo-style logging output: 0 has no output, N means an output every Nth cost function evaluation
     *
     * @return a tuple of the new value of x, the fitness of that x, and a status flag of optgra
     */
    std::tuple<std::vector<double>, std::vector<double>, int> sensitivity_update_constraint_delta(sensitivity_state state_tuple,
                                                                                                  const std::vector<int> &variable_types,
                                                                                                  const std::vector<int> &constraint_types, vector<double> &delta,
                                                                                                  double max_distance_per_iteration = 10,            // VARMAX
                                                                                                  double perturbation_for_snd_order_derivatives = 1, // VARSND
                                                                                                  std::vector<double> variable_scaling_factors = {},
                                                                                                  int log_level = 1,
                                                                                                  int verbosity = 0)
    {

        optgra_raii raii_object(variable_types, constraint_types,
                                1,                                      // max_iterations, // MAXITE
                                1,                                      // max_correction_iterations, // CORITE
                                max_distance_per_iteration,             // VARMAX
                                perturbation_for_snd_order_derivatives, // VARSND
                                {},                                     // convergence_thresholds,
                                variable_scaling_factors,
                                {}, // constraint_priorities,
                                {}, // variable_names,
                                {}, // constraint_names,
                                2,  // optimization_method, // OPTMET
                                0,  // VARDER
                                {},
                                log_level,
                                verbosity);

        raii_object.set_sensitivity_state_data(state_tuple);

        return raii_object.sensitivity_update_delta(delta);
    }

}
