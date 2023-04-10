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

#include <limbo/acqui.hpp>
#include <limbo/bayes_opt/boptimizer.hpp>
#include <limbo/init.hpp>
#include <limbo/tools/macros.hpp>

using namespace limbo;

namespace {
    struct Params {
        struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
            BO_PARAM(bool, stats_enabled, false);
        };

        struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
        };

        struct stop_maxiterations {
            BO_PARAM(int, iterations, 0);
        };

        struct kernel : public defaults::kernel {
            BO_PARAM(double, noise, 0.01);
        };

        struct kernel_maternfivehalves : public defaults::kernel_maternfivehalves {
            BO_PARAM(double, sigma_sq, 1);
            BO_PARAM(double, l, 0.25);
        };

        struct acqui_ucb : public defaults::acqui_ucb {
        };

#ifdef USE_NLOPT
        struct opt_nloptnograd : public defaults::opt_nloptnograd {
        };
#elif defined(USE_LIBCMAES)
        struct opt_cmaes : public defaults::opt_cmaes {
        };
#else
        struct opt_gridsearch : public defaults::opt_gridsearch {
        };
#endif
    };

    struct fit_eval {
        BO_PARAM(size_t, dim_in, 2);
        BO_PARAM(size_t, dim_out, 1);

        Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
        {
            double res = 0;
            for (int i = 0; i < x.size(); i++)
                res += 1 - (x[i] - 0.3) * (x[i] - 0.3) + sin(10 * x[i]) * 0.2;
            return tools::make_vector(res);
        }
    };
}
TEST(Limbo_Init_Functions, no_init)
{
    std::cout << "NoInit" << std::endl;
    using Model_t =  model::GP<kernel::MaternFiveHalves<limbo::defaults::kernel, limbo::defaults::kernel_maternfivehalves>>;
    using Acqui_t = acqui::UCB<Params::acqui_ucb, Model_t>;
    using Init_t = init::NoInit;
    using Opt_t = bayes_opt::BOptimizer<Params, Model_t, Acqui_t, Init_t>;

    Opt_t opt;
    opt.optimize(fit_eval());
    ASSERT_TRUE(opt.observations().size() == 0);
    ASSERT_TRUE(opt.samples().size() == 0);
}

struct RandLHSParams : public Params {
    struct init_lhs {
        BO_PARAM(int, samples, 10);
    };
};

TEST(Limbo_Init_Functions, random_lhs)
{
    std::cout << "LHS" << std::endl;

    using Model_t =  model::GP<kernel::MaternFiveHalves<limbo::defaults::kernel, limbo::defaults::kernel_maternfivehalves>>;
    using Acqui_t = acqui::UCB<RandLHSParams::acqui_ucb, Model_t>;
    using Init_t = init::LHS<RandLHSParams::init_lhs>;
    using Opt_t = bayes_opt::BOptimizer<RandLHSParams, Model_t, Acqui_t, Init_t>;

    Opt_t opt;
    opt.optimize(fit_eval());
    ASSERT_TRUE(opt.observations().size() == 10);
    ASSERT_TRUE(opt.samples().size() == 10);
    for (size_t j = 0; j < opt.samples().size() - 1; ++j) {
        const Eigen::VectorXd& x = opt.samples()[j];
        std::cout << x.transpose() << std::endl;
        for (int i = 0; i < x.size(); ++i) {
            ASSERT_TRUE(x[i] >= 0);
            ASSERT_TRUE(x[i] <= 1);
            ASSERT_TRUE(i == 0 || x[i] != x[0]);
        }
    }
}

struct RandSamplParams : public Params {
    struct init_randomsampling {
        BO_PARAM(int, samples, 10);
    };
};

TEST(Limbo_Init_Functions, random_sampling)
{
    std::cout << "RandomSampling" << std::endl;


    using Model_t =  model::GP<kernel::MaternFiveHalves<limbo::defaults::kernel, limbo::defaults::kernel_maternfivehalves>>;
    using Acqui_t = acqui::UCB<RandSamplParams::acqui_ucb, Model_t>;
    using Init_t = init::RandomSampling<RandSamplParams::init_randomsampling>;
    using Opt_t = bayes_opt::BOptimizer<RandSamplParams, Model_t, Acqui_t, Init_t>;

    Opt_t opt;
    opt.optimize(fit_eval());
    ASSERT_TRUE(opt.observations().size() == 10);
    ASSERT_TRUE(opt.samples().size() == 10);
    for (size_t j = 0; j < opt.samples().size() - 1; ++j) {
        const Eigen::VectorXd& x = opt.samples()[j];
        std::cout << x.transpose() << std::endl;
        for (int i = 0; i < x.size(); ++i) {
            ASSERT_TRUE(x[i] >= 0);
            ASSERT_TRUE(x[i] <= 1);
            ASSERT_TRUE(i == 0 || x[i] != x[0]);
        }
    }
}

struct RandSamplGridParams : public Params {
    struct init_randomsamplinggrid {
        BO_PARAM(int, samples, 10);
        BO_PARAM(int, bins, 4);
    };
};

TEST(Limbo_Init_Functions, random_sampling_grid)
{
    std::cout << "RandomSamplingGrid" << std::endl;
   

    using Model_t =  model::GP<kernel::MaternFiveHalves<limbo::defaults::kernel, limbo::defaults::kernel_maternfivehalves>>;
    using Acqui_t = acqui::UCB<RandSamplGridParams::acqui_ucb, Model_t>;
    using Init_t = init::RandomSamplingGrid<RandSamplGridParams::init_randomsamplinggrid>;
    using Opt_t = bayes_opt::BOptimizer<RandSamplGridParams, Model_t, Acqui_t, Init_t>;

    Opt_t opt;
    opt.optimize(fit_eval());
    ASSERT_TRUE(opt.observations().size() == 10);
    ASSERT_TRUE(opt.samples().size() == 10);
    for (size_t j = 0; j < opt.samples().size() - 1; ++j) {
        const Eigen::VectorXd& x = opt.samples()[j];
        std::cout << x.transpose() << std::endl;
        for (int i = 0; i < x.size(); ++i) {
            ASSERT_TRUE(x[i] >= 0);
            ASSERT_TRUE(x[i] <= 1);
            ASSERT_TRUE(x[i] == 0 || x[i] == 0.25 || x[i] == 0.5 || x[i] == 0.75 || x[i] == 1.0);
        }
    }
}

struct GridSamplParams : public Params {
    struct init_gridsampling {
        BO_PARAM(int, bins, 4);
    };
};

TEST(Limbo_Init_Functions, grid_sampling)
{
    std::cout << "GridSampling" << std::endl;


    using Model_t =  model::GP<kernel::MaternFiveHalves<limbo::defaults::kernel, limbo::defaults::kernel_maternfivehalves>>;
    using Acqui_t = acqui::UCB<GridSamplParams::acqui_ucb, Model_t>;
    using Init_t = init::GridSampling<GridSamplParams::init_gridsampling>;
    using Opt_t = bayes_opt::BOptimizer<GridSamplParams, Model_t, Acqui_t, Init_t>;

    Opt_t opt;
    opt.optimize(fit_eval());
    std::cout << opt.observations().size() << std::endl;
    ASSERT_TRUE(opt.observations().size() == 25);
    ASSERT_TRUE(opt.samples().size() == 25);
    for (size_t j = 0; j < opt.samples().size() - 1; ++j) {
        const Eigen::VectorXd& x = opt.samples()[j];
        std::cout << x.transpose() << std::endl;
        for (int i = 0; i < x.size(); ++i) {
            ASSERT_TRUE(x[i] >= 0);
            ASSERT_TRUE(x[i] <= 1);
            ASSERT_TRUE(x[i] == 0 || x[i] == 0.25 || x[i] == 0.5 || x[i] == 0.75 || x[i] == 1.0);
        }
    }
}
