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
#include <limbo/opt/adam.hpp>
#include <limbo/opt/chained.hpp>
#include <limbo/opt/cmaes.hpp>
#include <limbo/opt/gradient_ascent.hpp>
#include <limbo/opt/grid_search.hpp>
#include <limbo/opt/parallel_repeater.hpp>
#include <limbo/opt/random_point.hpp>
#include <limbo/opt/rprop.hpp>
#include <limbo/tools/macros.hpp>
#ifdef USE_TBB
#include <tbb/global_control.h>
#endif
using namespace limbo;

namespace {
    struct Params {
        struct opt_gridsearch {
            BO_PARAM(int, bins, 20);
        };

        struct opt_parallelrepeater {
            BO_PARAM(int, repeats, 2);
            BO_PARAM(double, epsilon, 0.1);
        };

        struct opt_rprop : public defaults::opt_rprop {
            BO_PARAM(int, iterations, 150);
        };

        struct opt_gradient_ascent : public defaults::opt_gradient_ascent {
            BO_PARAM(int, iterations, 150);
            BO_PARAM(double, alpha, 0.1);
        };

        struct opt_adam : public defaults::opt_adam {
            BO_PARAM(int, iterations, 150);
            BO_PARAM(double, alpha, 0.1);
        };
    };
}

// test with a standard function
int monodim_calls = 0;
opt::eval_t acqui_mono(const Eigen::VectorXd& v, bool eval_grad)
{
    assert(!eval_grad);
    monodim_calls++;
    return opt::no_grad(3 * v(0) + 5);
}

// test with a functor
int bidim_calls = 0;
struct FakeAcquiBi {
    opt::eval_t operator()(const Eigen::VectorXd& v, bool eval_grad) const
    {
        assert(!eval_grad);
        bidim_calls++;
        return opt::no_grad(3 * v(0) + 5 - 2 * v(1) - 5 * v(1) + 2);
    }
};

// test with gradient
int simple_calls = 0;
bool check_grad = false;
std::vector<Eigen::VectorXd> starting_points;
opt::eval_t simple_func(const Eigen::VectorXd& v, bool eval_grad)
{
    assert(!check_grad || eval_grad);
    simple_calls++;
    starting_points.push_back(v);
    return {-(v(0) * v(0) + 2. * v(0)), limbo::tools::make_vector(-(2 * v(0) + 2.))};
}

TEST(Limbo_Optimizers, random_mono_dim)
{
    using namespace limbo;

    opt::RandomPoint optimizer = opt::RandomPoint::create(1);

    monodim_calls = 0;
    for (int i = 0; i < 1000; i++) {
        Eigen::VectorXd best_point = optimizer.optimize(acqui_mono, Eigen::VectorXd::Constant(1, 0.5), true);
        ASSERT_EQ(best_point.size(), 1);
        ASSERT_TRUE(best_point(0) > 0 || std::abs(best_point(0)) < 1e-7);
        ASSERT_TRUE(best_point(0) < 1 || std::abs(best_point(0) - 1) < 1e-7);
    }
}

TEST(Limbo_Optimizers, random_bi_dim)
{
    using namespace limbo;

    opt::RandomPoint optimizer = opt::RandomPoint::create(1);

    bidim_calls = 0;
    for (int i = 0; i < 1000; i++) {
        Eigen::VectorXd best_point = optimizer.optimize(FakeAcquiBi(), Eigen::VectorXd::Constant(2, 0.5), true);
        ASSERT_EQ(best_point.size(), 2);
        ASSERT_TRUE(best_point(0) > 0 || std::abs(best_point(0)) < 1e-7);
        ASSERT_TRUE(best_point(0) < 1 || std::abs(best_point(0) - 1) < 1e-7);
        ASSERT_TRUE(best_point(1) > 0 || std::abs(best_point(1)) < 1e-7);
        ASSERT_TRUE(best_point(1) < 1 || std::abs(best_point(1) - 1) < 1e-7);
    }
}

TEST(Limbo_Optimizers, grid_search_mono_dim)
{
    using namespace limbo;

    opt::GridSearch<Params::opt_gridsearch> optimizer;

    monodim_calls = 0;
    Eigen::VectorXd best_point = optimizer.optimize(acqui_mono, Eigen::VectorXd::Constant(1, 0.5), true);

    ASSERT_EQ(best_point.size(), 1);
    ASSERT_NEAR(best_point(0), 1, 0.0001);
    ASSERT_EQ(monodim_calls, Params::opt_gridsearch::bins() + 1);
}

TEST(Limbo_Optimizers, grid_search_bi_dim)
{
    using namespace limbo;

    opt::GridSearch<Params::opt_gridsearch> optimizer;

    bidim_calls = 0;
    Eigen::VectorXd best_point = optimizer.optimize(FakeAcquiBi(), Eigen::VectorXd::Constant(2, 0.5), true);

    ASSERT_EQ(best_point.size(), 2);
    ASSERT_NEAR(best_point(0), 1, 0.0001);
    ASSERT_LE(best_point(1), 0.000001);
    // TO-DO: Maybe alter a little grid search so not to call more times the utility function
    ASSERT_EQ(bidim_calls, (Params::opt_gridsearch::bins() + 1) * (Params::opt_gridsearch::bins() + 1) + 21);
}

TEST(Limbo_Optimizers, gradient)
{
    using namespace limbo;

    opt::Rprop<Params::opt_rprop> optimizer;

    simple_calls = 0;
    check_grad = true;
    Eigen::VectorXd best_point = optimizer.optimize(simple_func, Eigen::VectorXd::Constant(1, 2.0), false);
    ASSERT_EQ(best_point.size(), 1);
    ASSERT_TRUE(std::abs(best_point(0) + 1.) < 1e-3);
    ASSERT_EQ(simple_calls, Params::opt_rprop::iterations());
}


struct MomentumParams {
    struct opt_gradient_ascent : public defaults::opt_gradient_ascent {
        BO_PARAM(int, iterations, 150);
        BO_PARAM(double, alpha, 0.1);
        BO_PARAM(double, gamma, 0.8);
    };
};

struct NesterovParams {
    struct opt_gradient_ascent : public defaults::opt_gradient_ascent {
        BO_PARAM(int, iterations, 150);
        BO_PARAM(double, alpha, 0.1);
        BO_PARAM(double, gamma, 0.8);
        BO_PARAM(bool, nesterov, true);
    };
};

TEST(Limbo_Optimizers, classic_optimizers)
{
    using namespace limbo;

    opt::Rprop<Params::opt_rprop> rprop;
    opt::Adam<Params::opt_adam> adam;
    opt::GradientAscent<Params::opt_gradient_ascent> gradient_ascent;
    opt::GradientAscent<MomentumParams::opt_gradient_ascent> gradient_ascent_momentum;
    opt::GradientAscent<NesterovParams::opt_gradient_ascent> gradient_ascent_nesterov;

    simple_calls = 0;
    check_grad = true;
    Eigen::VectorXd best_point = rprop.optimize(simple_func, Eigen::VectorXd::Constant(1, 2.0), false);
    ASSERT_EQ(best_point.size(), 1);
    ASSERT_TRUE(std::abs(best_point(0) + 1.) < 1e-3);
    ASSERT_EQ(simple_calls, Params::opt_rprop::iterations());

    double best_rprop = best_point(0);

    simple_calls = 0;
    check_grad = true;
    best_point = gradient_ascent.optimize(simple_func, Eigen::VectorXd::Constant(1, 2.0), false);
    ASSERT_EQ(best_point.size(), 1);
    ASSERT_TRUE(std::abs(best_point(0) + 1.) < 1e-3);
    ASSERT_EQ(simple_calls, Params::opt_gradient_ascent::iterations());

    double best_gradient_ascent = best_point(0);

    simple_calls = 0;
    check_grad = true;
    best_point = gradient_ascent_momentum.optimize(simple_func, Eigen::VectorXd::Constant(1, 2.0), false);
    ASSERT_EQ(best_point.size(), 1);
    ASSERT_TRUE(std::abs(best_point(0) + 1.) < 1e-3);
    ASSERT_EQ(simple_calls, MomentumParams::opt_gradient_ascent::iterations());

    double best_gradient_ascent_momentum = best_point(0);

    simple_calls = 0;
    check_grad = true;
    best_point = gradient_ascent_nesterov.optimize(simple_func, Eigen::VectorXd::Constant(1, 2.0), false);
    ASSERT_EQ(best_point.size(), 1);
    ASSERT_TRUE(std::abs(best_point(0) + 1.) < 1e-3);
    ASSERT_EQ(simple_calls, NesterovParams::opt_gradient_ascent::iterations());

    double best_gradient_ascent_nesterov = best_point(0);

    simple_calls = 0;
    check_grad = true;
    best_point = adam.optimize(simple_func, Eigen::VectorXd::Constant(1, 2.0), false);
    ASSERT_EQ(best_point.size(), 1);
    ASSERT_TRUE(std::abs(best_point(0) + 1.) < 1e-3);
    ASSERT_EQ(simple_calls, Params::opt_adam::iterations());

    double best_adam = best_point(0);

    ASSERT_TRUE(std::abs(best_rprop - best_gradient_ascent) < 1e-3);
    ASSERT_TRUE(std::abs(best_rprop - best_gradient_ascent_momentum) < 1e-3);
    ASSERT_TRUE(std::abs(best_rprop - best_gradient_ascent_nesterov) < 1e-3);
    ASSERT_TRUE(std::abs(best_rprop - best_adam) < 1e-3);
}

TEST(Limbo_Optimizers, parallel_repeater)
{
#ifdef USE_TBB
    tbb::global_control(tbb::global_control::max_allowed_parallelism, 1);
#endif
    using namespace limbo;

    opt::ParallelRepeater<Params::opt_parallelrepeater, opt::Rprop<Params::opt_rprop>> optimizer;

    simple_calls = 0;
    check_grad = false;
    starting_points.clear();
    Eigen::VectorXd best_point = optimizer.optimize(simple_func, Eigen::VectorXd::Constant(1, 2.0), false);
    ASSERT_EQ(best_point.size(), 1);
    ASSERT_TRUE(std::abs(best_point(0) + 1.) < 1e-3);
    ASSERT_EQ(simple_calls, Params::opt_parallelrepeater::repeats() * Params::opt_rprop::iterations() + Params::opt_parallelrepeater::repeats());
    ASSERT_EQ(starting_points.size(), simple_calls);
    ASSERT_TRUE(starting_points[0](0) >= 2. - Params::opt_parallelrepeater::epsilon() && starting_points[0](0) <= 2. + Params::opt_parallelrepeater::epsilon());
    ASSERT_TRUE(starting_points[Params::opt_rprop::iterations() + 1](0) >= 2. - Params::opt_parallelrepeater::epsilon() && starting_points[Params::opt_rprop::iterations() + 1](0) <= 2. + Params::opt_parallelrepeater::epsilon());
#ifdef USE_TBB
    tools::par::init();
#endif
}

TEST(Limbo_Optimizers, chained)
{
    using namespace limbo;

    using opt_1_t = opt::GridSearch<Params::opt_gridsearch>;
    using opt_2_t = opt::RandomPoint;
    using opt_3_t = opt::GridSearch<Params::opt_gridsearch>;
    using opt_4_t = opt::GridSearch<Params::opt_gridsearch>;
    auto optimizer = opt::Chained<opt_1_t, opt_2_t, opt_3_t, opt_4_t>::create(1);

    monodim_calls = 0;
    Eigen::VectorXd best_point = optimizer.optimize(acqui_mono, Eigen::VectorXd::Constant(1, 0.5), true);

    ASSERT_EQ(best_point.size(), 1);
    ASSERT_TRUE(best_point(0) > 0 || std::abs(best_point(0)) < 1e-7);
    ASSERT_TRUE(best_point(0) < 1 || std::abs(best_point(0) - 1) < 1e-7);
    ASSERT_EQ(monodim_calls, (Params::opt_gridsearch::bins() + 1) * 3);
}
