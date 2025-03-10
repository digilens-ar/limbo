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

#include <gtest/gtest.h>
#include <limbo/limbo.hpp>

using namespace limbo;

namespace {
    struct Params {

        struct opt_rprop : public defaults::opt_rprop {
        };

#ifdef USE_NLOPT
        struct opt_nloptnograd : public defaults::opt_nloptnograd {
        };
#elif defined(USE_LIBCMAES)
        struct opt_cmaes : public defaults::opt_cmaes {
        };
#endif
        struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
            BO_DYN_PARAM(int, hp_period);
            BO_PARAM(bool, stats_enabled, false);
        };

        struct stop_maxiterations : defaults::stop_maxiterations {
            BO_PARAM(int, iterations, 200);
        };

        struct kernel : public defaults::kernel {
            BO_PARAM(double, noise, 1e-8);
        };

        struct kernel_exp : public defaults::kernel_exp {
            BO_PARAM(double, l, 0.2);
            BO_PARAM(double, sigma_sq, 0.25);
        };

        struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
            BO_PARAM(double, sigma_sq, 0.25);
        };

        struct acqui_ucb {
            BO_PARAM(double, kappa, 1.0);
        };

        struct acqui_ei {
            BO_PARAM(double, jitter, 0.001);
        };

        struct init_randomsampling {
            BO_PARAM(int, samples, 50);
        };

        struct opt_parallelrepeater : defaults::opt_parallelrepeater {
        };
    };

    BO_DECLARE_DYN_PARAM(int, Params::bayes_opt_boptimizer, hp_period);

    template <typename Params>
    struct eval2 {
        BO_PARAM(size_t, dim_in, 2);

        std::tuple<EvaluationStatus, double> operator()(const Eigen::VectorXd& x) const
        {
            Eigen::Vector2d opt(0.25, 0.75);
            return { OK, -(x - opt).squaredNorm() };
        }
    };

#ifdef USE_LIBCMAES
    template <typename Params, int obs_size = 1>
    struct eval_unbounded {
        BO_PARAM(size_t, dim_in, 1);
        BO_PARAM(size_t, dim_out, obs_size);

        Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
        {
            return Eigen::VectorXd{ {-std::pow(x(0) - 2.5, 2.0)} };
        }
    };
#endif
}

struct InheritParameters {
    struct stop_maxiterations : defaults::stop_maxiterations {
        BO_PARAM(int, iterations, 1);
    };
};

TEST(Limbo_Boptimizer, bo_inheritance)
{
    using namespace limbo;

    Params::bayes_opt_boptimizer::set_hp_period(-1);

    using Kernel_t = kernel::Exp<Params::kernel, Params::kernel_exp>;
#ifdef USE_NLOPT
    using AcquiOpt_t = opt::NLOptNoGrad<Params::opt_nloptnograd, nlopt::GN_DIRECT_L_RAND>;
#else
    using AcquiOpt_t = opt::Cmaes<Params>;
#endif
    using Stop_t = boost::fusion::vector<stop::MaxIterations<InheritParameters::stop_maxiterations>>;
    using Mean_t = mean::Data;
    using Stat_t = boost::fusion::vector<limbo::stat::Samples, limbo::stat::Observations>;
    using Init_t = init::NoInit;
    using GP_t = model::GaussianProcess<Kernel_t, Mean_t>;
    using Acqui_t = acqui::UCB<Params::acqui_ucb, GP_t>;

    bayes_opt::BOptimizer<Params, GP_t, Acqui_t, Init_t, Stop_t, Stat_t, AcquiOpt_t> opt(eval2<Params>::dim_in());
    opt.optimize(eval2<Params>());

    ASSERT_TRUE(opt.total_iterations() == 1);
}

#ifdef USE_LIBCMAES
TEST(Limbo_Boptimizer, bo_unbounded)
{
    using namespace limbo;

    struct Parameters {

        struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
            BO_PARAM(int, hp_period, -1);
            BO_PARAM(bool, stats_enabled, false);
            BO_PARAM(bool, bounded, false);
        };

        struct opt_cmaes : public defaults::opt_cmaes {
        };
    };

    using Kernel_t = kernel::Exp<Params>;
    using AcquiOpt_t = opt::Cmaes<Parameters>;
    using Stop_t = boost::fusion::vector<stop::MaxIterations<Params>>;
    using Mean_t = mean::NullFunction<Params>;
    using Stat_t = boost::fusion::vector<stat::ConsoleSummary<Params>>;
    using Init_t = init::RandomSampling<Params>;
    using GP_t = model::GaussianProcess<Params, Kernel_t, Mean_t>;
    using Acqui_t = acqui::UCB<Params, GP_t>;

    bayes_opt::BOptimizer<Parameters, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>> opt;
    opt.optimize(eval_unbounded<Params>());

    ASSERT_NEAR(opt.best_sample()(0), 2.5, 10);
}
#endif

TEST(Limbo_Boptimizer, bo_gp)
{
    using namespace limbo;

    Params::bayes_opt_boptimizer::set_hp_period(-1);

    using Kernel_t = kernel::Exp<Params::kernel, Params::kernel_exp>;
#ifdef USE_NLOPT
    using AcquiOpt_t = opt::NLOptNoGrad<Params::opt_nloptnograd, nlopt::GN_DIRECT_L_RAND>;
#else
    using AcquiOpt_t = opt::Cmaes<Params>;
#endif
    using Stop_t = boost::fusion::vector<stop::MaxIterations<Params::stop_maxiterations>>;
    using Mean_t = mean::Data;
    using Stat_t = boost::fusion::vector<limbo::stat::Samples, limbo::stat::Observations>;
    using Init_t = init::RandomSampling<Params::init_randomsampling>;
    using GP_t = model::GaussianProcess<Kernel_t, Mean_t>;
    using Acqui_t = acqui::EI<Params::acqui_ei, GP_t>;

    bayes_opt::BOptimizer<Params, GP_t, Acqui_t, Init_t, Stop_t, Stat_t, AcquiOpt_t> opt(eval2<Params>::dim_in());
    opt.optimize(eval2<Params>());

    Eigen::VectorXd sol(2);
    sol << 0.25, 0.75;
    ASSERT_TRUE((sol - opt.model().best_observation().second).squaredNorm() < 1e-3);
}

TEST(Limbo_Boptimizer, bo_gp_auto)
{
    using namespace limbo;

    Params::bayes_opt_boptimizer::set_hp_period(50);

    using Kernel_t = kernel::SquaredExpARD<Params::kernel, Params::kernel_squared_exp_ard>;
#ifdef USE_NLOPT
    using AcquiOpt_t = opt::NLOptNoGrad<Params::opt_nloptnograd, nlopt::GN_DIRECT_L_RAND>;
#else
    using AcquiOpt_t = opt::Cmaes<Params>;
#endif
    using Stop_t = boost::fusion::vector<stop::MaxIterations<Params::stop_maxiterations>>;
    using Mean_t = mean::Data;
    using Stat_t = boost::fusion::vector<limbo::stat::Samples, limbo::stat::Observations>;
    using Init_t = init::RandomSampling<Params::init_randomsampling>;
    using GP_t = model::GaussianProcess<Kernel_t, Mean_t, model::gp::KernelLFOpt<opt::Rprop<Params::opt_rprop>>>;
    using Acqui_t = acqui::UCB<Params::acqui_ucb, GP_t>;

    bayes_opt::BOptimizer<Params, GP_t, Acqui_t, Init_t, Stop_t, Stat_t, AcquiOpt_t> opt(eval2<Params>::dim_in());
    opt.optimize(eval2<Params>());

    Eigen::VectorXd sol(2);
    sol << 0.25, 0.75;
    ASSERT_TRUE((sol - opt.model().best_observation().second).squaredNorm() < 1e-3);
}

TEST(Limbo_Boptimizer, bo_gp_mean)
{
    using namespace limbo;

    Params::bayes_opt_boptimizer::set_hp_period(50);

    using Kernel_t = kernel::SquaredExpARD<Params::kernel, Params::kernel_squared_exp_ard>;
#ifdef USE_NLOPT
    using AcquiOpt_t = opt::NLOptNoGrad<Params::opt_nloptnograd, nlopt::GN_DIRECT_L_RAND>;
#else
    using AcquiOpt_t = opt::Cmaes<Params>;
#endif
    using Stop_t = boost::fusion::vector<stop::MaxIterations<Params::stop_maxiterations>>;
    using Mean_t = mean::FunctionARD<mean::Data>;
    using Stat_t = boost::fusion::vector<limbo::stat::Samples, limbo::stat::Observations>;
    using Init_t = init::RandomSampling<Params::init_randomsampling>;
    using GP_t = model::GaussianProcess<Kernel_t, Mean_t, model::gp::MeanLFOpt<opt::Rprop<Params::opt_rprop>>>;
    using Acqui_t = acqui::UCB<Params::acqui_ucb, GP_t>;

    bayes_opt::BOptimizer<Params, GP_t, Acqui_t, Init_t, Stop_t, Stat_t, AcquiOpt_t> opt(eval2<Params>::dim_in());
    opt.optimize(eval2<Params>());

    Eigen::VectorXd sol(2);
    sol << 0.25, 0.75;
    ASSERT_TRUE((sol - opt.model().best_observation().second).squaredNorm() < 1e-3);
}
