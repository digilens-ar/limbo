//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/limbo
//| Documentation: http://www.resibots.eu/limbo
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
#include <chrono>
#include <iostream>

#include <limbo/limbo.hpp>

#include "testfunctions.hpp"
#include "benchmark/benchmark.h"


#define LIMBO_DEF // Select the type of Bayes optimizer to test with


using namespace limbo;

struct Params {
    struct bayes_opt_boptimizer : defaults::bayes_opt_boptimizer {
#if defined(LIMBO_DEF_HPOPT)
        BO_PARAM(int, hp_period, 50);
#else
        BO_PARAM(int, hp_period, -1);
#endif
        BO_PARAM(bool, stats_enabled, false);
    };
    struct stop_maxiterations : defaults::stop_maxiterations {
        BO_PARAM(int, iterations, 190);
    };
    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 1e-10);
    };
    struct kernel_exp : public defaults::kernel_exp {
    };
    struct kernel_maternfivehalves {
        BO_PARAM(double, sigma_sq, 1);
        BO_PARAM(double, l, 1);
    };
    struct acqui_ucb : public defaults::acqui_ucb {
        BO_PARAM(double, kappa, 0.125);
    };
    struct acqui_ei : public defaults::acqui_ei {
    };
    struct init_randomsampling {
        BO_PARAM(int, samples, 10);
    };
    struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
    };
    struct mean_constant {
        BO_PARAM(double, constant, 1);
    };
    struct opt_rprop : public defaults::opt_rprop {
        BO_PARAM(double, eps_stop, 1e-6);
    };
    struct opt_parallelrepeater : public defaults::opt_parallelrepeater {
    };
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
        BO_PARAM(double, fun_tolerance, 1e-6);
        BO_PARAM(double, xrel_tolerance, 1e-6);
    };
#ifdef USE_LIBCMAES
    struct opt_cmaes : public defaults::opt_cmaes {
        BO_PARAM(int, max_fun_evals, 500);
        BO_PARAM(double, fun_tolerance, 1e-6);
        BO_PARAM(double, xrel_tolerance, 1e-6);
    };
#endif
};

struct DirectParams {
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
        BO_DYN_PARAM(int, iterations);
    };
};

struct BobyqaParams {
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
        BO_DYN_PARAM(int, iterations);
    };
};
struct BobyqaParams_HP {
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
        BO_DYN_PARAM(int, iterations);
    };
};

BO_DECLARE_DYN_PARAM(int, DirectParams::opt_nloptnograd, iterations);
BO_DECLARE_DYN_PARAM(int, BobyqaParams::opt_nloptnograd, iterations);
BO_DECLARE_DYN_PARAM(int, BobyqaParams_HP::opt_nloptnograd, iterations);

template <concepts::BayesOptimizer Optimizer, TestFunction Function>
void optimize(benchmark::State& state)
{
    int iters_base = 250;
    DirectParams::opt_nloptnograd::set_iterations(static_cast<int>(iters_base * Function::dim_in() * 0.9));
    BobyqaParams::opt_nloptnograd::set_iterations(iters_base * Function::dim_in() - DirectParams::opt_nloptnograd::iterations());

    BobyqaParams_HP::opt_nloptnograd::set_iterations(10 * Function::dim_in() * Function::dim_in());

    double bestObs = 0;
    Eigen::VectorXd bestSample;
    double accuracy = 100;
    for (auto _ : state) {
        srand(time(NULL));
        Optimizer opt(Function::dim_in());
        Benchmark<Function> target;
        opt.optimize(target);
        std::tie(bestObs, bestSample) = opt.model().best_observation();
        accuracy = target.accuracy(bestObs);
    }
    std::cout << "Result: " << std::fixed << bestSample.transpose() << " -> " << bestObs << std::endl;
    std::cout << "Smallest difference: " << accuracy << std::endl;
}


// limbo default parameters
#ifdef LIMBO_DEF
using Opt_t = bayes_opt::BOptimizer<Params>;
#elif defined(LIMBO_DEF_HPOPT)
using Opt_t = bayes_opt::BOptimizerHPOpt<Params>;

// benchmark different optimization algorithms
#elif defined(OPT_CMAES)
using AcquiOpt_t = opt::Cmaes<Params>;
using Opt_t = bayes_opt::BOptimizer<Params, acquiopt<AcquiOpt_t>>;
#elif defined(OPT_DIRECT)
using AcquiOpt_t = opt::Chained<Params, opt::NLOptNoGrad<DirectParams, nlopt::GN_DIRECT_L>, opt::NLOptNoGrad<BobyqaParams, nlopt::LN_BOBYQA>>;
using Opt_t = bayes_opt::BOptimizer<Params, acquiopt<AcquiOpt_t>>;

//benchmark different acquisition functions
#elif defined(ACQ_UCB)
using GP_t = model::GP<Params>;
using Acqui_t = acqui::UCB<Params, GP_t>;
using Opt_t = bayes_opt::BOptimizer<Params, acquifun<Acqui_t>>;
#elif defined(ACQ_EI)
using GP_t = model::GP<Params>;
using Acqui_t = acqui::EI<Params, GP_t>;
using Opt_t = bayes_opt::BOptimizer<Params, acquifun<Acqui_t>>;
#else
#error "Unknown variant in benchmark"
#endif

BENCHMARK(optimize<Opt_t, BraninNormalized>)->Unit(benchmark::kMillisecond)->MinTime(6);
BENCHMARK(optimize<Opt_t, Hartmann6>)->Unit(benchmark::kMillisecond)->MinTime(6);
BENCHMARK(optimize<Opt_t, Hartmann3>)->Unit(benchmark::kMillisecond)->MinTime(6);
BENCHMARK(optimize<Opt_t, Rastrigin>)->Unit(benchmark::kMillisecond)->MinTime(6);
BENCHMARK(optimize<Opt_t, Sphere>)->Unit(benchmark::kMillisecond)->MinTime(6);
BENCHMARK(optimize<Opt_t, Ellipsoid>)->Unit(benchmark::kMillisecond)->MinTime(6);
BENCHMARK(optimize<Opt_t, GoldsteinPrice>)->Unit(benchmark::kMillisecond)->MinTime(6);
BENCHMARK(optimize<Opt_t, SixHumpCamel>)->Unit(benchmark::kMillisecond)->MinTime(6);


