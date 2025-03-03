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
#include <limbo/opt/nlopt_grad.hpp>
#include <limbo/opt/nlopt_no_grad.hpp>

using namespace limbo;

namespace {
    struct Params {
        struct opt_nloptgrad : public defaults::opt_nloptgrad {
            BO_PARAM(int, iterations, 200);
        };

        struct opt_nloptnograd : public defaults::opt_nloptnograd {
            BO_PARAM(int, iterations, 200);
        };
    };

    opt::eval_t my_function(const Eigen::VectorXd& params, bool eval_grad)
    {
        double v = -params(0) * params(0) - params(1) * params(1);
        if (!eval_grad)
            return opt::no_grad(v);
        Eigen::VectorXd grad(2);
        grad(0) = -2 * params(0);
        grad(1) = -2 * params(1);
        return { v, grad };
    }

    opt::eval_t my_constraint(const Eigen::VectorXd& params, bool eval_grad)
    {
        double v = params(0) + 3. * params(1) - 10.;
        if (!eval_grad)
            return opt::no_grad(v);
        Eigen::VectorXd grad(2);
        grad(0) = 1.;
        grad(1) = 3.;
        return { v, grad };
    }

    opt::eval_t my_inequality_constraint(const Eigen::VectorXd& params, bool eval_grad)
    {
        double v = -params(0) - 3. * params(1) + 10.;
        if (!eval_grad)
            return opt::no_grad(v);
        Eigen::VectorXd grad(2);
        grad(0) = -1.;
        grad(1) = -3.;
        return { v, grad };
    }
}

TEST(Limbo_NLOpt, nlopt_grad_simple)
{
	auto optimizer = opt::NLOptGrad<Params::opt_nloptgrad, nlopt::LD_MMA>::create(2);
    Eigen::VectorXd g = optimizer.optimize(my_function, tools::random_vector(2), std::nullopt);

    ASSERT_LE(g(0), 0.00000001);
    ASSERT_LE(g(1), 0.00000001);
}

TEST(Limbo_NLOpt, nlopt_no_grad_simple)
{
    auto optimizer = opt::NLOptNoGrad<Params::opt_nloptnograd, nlopt::LN_COBYLA>::create(2);
    Eigen::VectorXd best(2);
    best << 1, 1;
    size_t N = 10;
    for (size_t i = 0; i < N; i++) {
        Eigen::VectorXd g = optimizer.optimize(my_function, tools::random_vector(2), std::nullopt);
        if (g.norm() < best.norm()) {
            best = g;
        }
    }

    ASSERT_LE(best(0), 0.00000001);
    ASSERT_LE(best(1), 0.00000001);
}

TEST(Limbo_NLOpt, nlopt_no_grad_constraint)
{
    auto optimizer = opt::NLOptNoGrad<Params::opt_nloptnograd, nlopt::LN_COBYLA>::create(2);
    optimizer.add_equality_constraint(my_constraint);

    Eigen::VectorXd best = tools::random_vector(2).array() * 50.; // some random big value
    Eigen::VectorXd target(2);
    target << 1., 3.;
    size_t N = 10;
    for (size_t i = 0; i < N; i++) {
        Eigen::VectorXd g = optimizer.optimize(my_function, tools::random_vector(2), std::nullopt);
        if ((g - target).norm() < (best - target).norm()) {
            best = g;
        }
    }

    ASSERT_LE(std::abs(1. - best(0)), 0.000001);
    ASSERT_LE(std::abs(3. - best(1)), 0.000001);
}

// Loads a series of saved Bayesian models and evaluates the various NLOPT algorithms on them to see which one finds the highest maximum. In my testing the default of DIRECT_L_RAND always did best
// template<nlopt::algorithm alg>
// class NL: public opt::NLOptNoGrad<defaults::opt_nloptnograd, alg> {};
//
// TEST(Limbo_NLOpt, nlopt_comparison)
// {
//     struct MyParams
//     {
// 	    struct kernel_opt: defaults::kernel {};
//
// 	    struct kernel_squared_exp_ard : defaults::kernel_squared_exp_ard {};
//
// 	    struct acqui_ucb : defaults::acqui_ucb {};
//     };
//
//     using Model = model::GaussianProcess<kernel::SquaredExpARD<MyParams::kernel_opt, MyParams::kernel_squared_exp_ard>>;
//     using Acqui = acqui::UCB<MyParams::acqui_ucb, Model>;
//
// 	std::filesystem::path root = R"(C:\Users\NicholasAnthony\source\repos\wt_gui\external\WaveTracer\testing\optimizerTests\resources\DIA_highD\optimizations\hp_10_irpropplus_dual_1eneg3_unbound)";
//
//     std::ofstream f(root / "data.csv");
//     std::string header = "ISRES\tDIRECT_L_RAND\tORIG_DIRECT_L\tDIRECT\tDIRECT_L\tDIRECT_L_NOSCAL\tDIRECT_L_RAND_NOSCAL\tORIG_DIRECT\tESCH\tCRS2_LM\tAUG_LAG\n";
//     std::cout << header;
//     f << header;
//
//     for (auto const& entry : std::filesystem::directory_iterator(root))
//     {
//         if (!entry.is_directory() || !entry.path().filename().string().starts_with("modelArchive"))
//             continue;
//
//         auto m = Model::load(serialize::TextArchive(entry.path().string()));
//
//         Acqui acqui(m);
//         Eigen::VectorXd starting_point = tools::random_vector(m.dim_in(), true);
//         auto parameterBounds = std::vector<std::pair<double, double>>(m.dim_in(), std::make_pair(0.0, 1.0));
//
//
//         auto test = [&starting_point, &acqui, &parameterBounds](auto const& optimizer) -> double
//         {
//             double best;
//             Eigen::VectorXd new_sample = optimizer.optimize(
//                 [&](const Eigen::VectorXd& x, bool g) -> opt::eval_t { return acqui(x, g); },
//                 starting_point,
//                 parameterBounds,
//                 &best);
//             return best;
//         };
//
//
//         auto opt1 = NL<nlopt::GN_ISRES>::create(m.dim_in());
//         auto opt2 = NL<nlopt::GN_DIRECT_L_RAND>::create(m.dim_in());
//         auto opt3 = NL<nlopt::GN_ORIG_DIRECT_L>::create(m.dim_in());
//         auto opt4 = NL<nlopt::GN_DIRECT>::create(m.dim_in());
//         auto opt5 = NL<nlopt::GN_DIRECT_L>::create(m.dim_in());
//         auto opt6 = NL<nlopt::GN_DIRECT_L_NOSCAL>::create(m.dim_in());
//         auto opt7 = NL<nlopt::GN_DIRECT_L_RAND_NOSCAL>::create(m.dim_in());
//         auto opt8 = NL<nlopt::GN_ORIG_DIRECT>::create(m.dim_in());
//         auto opt9 = NL<nlopt::GN_ESCH>::create(m.dim_in());
//         auto opt10 = NL<nlopt::GN_CRS2_LM>::create(m.dim_in());
//         auto opt11 = NL<nlopt::AUGLAG>::create(m.dim_in());
//
//         auto tuple = std::make_tuple(opt1, opt2, opt3, opt4, opt5, opt6, opt7, opt8, opt9, opt10, opt11);
//         std::vector<double> results;
//         std::apply([&results, test](auto const& ... opts) 
//             {
// 				((results.push_back(test(opts))), ...);
//             }, tuple);
//
//         for (int i=0; i<results.size(); i++)
//         {
//             std::cout << results.at(i);
//             f << results.at(i);
//             if (i == results.size() - 1)
//             {
//                 std::cout << "\n";
//                 f << "\n";
//             }
//             else
//             {
//                 std::cout << "\t";
//                 f << "\t";
//             }
//         }
//     }
// }

//TODO this test crashed accessing a null value in NLOpt
// TEST(Limbo_NLOpt, nlopt_grad_constraint)
// {
//     opt::NLOptGrad<Params, nlopt::LD_AUGLAG_EQ> optimizer;
//     optimizer.initialize(2);
//     optimizer.add_inequality_constraint(my_inequality_constraint);
//
//     Eigen::VectorXd best = tools::random_vector(2).array() * 50.; // some random big value
//     Eigen::VectorXd target(2);
//     target << 1., 3.;
//     size_t N = 10;
//     for (size_t i = 0; i < N; i++) {
//         Eigen::VectorXd g = optimizer(my_function, tools::random_vector(2), false);
//         if ((g - target).norm() < (best - target).norm()) {
//             best = g;
//         }
//     }
//
//     ASSERT_LE(std::abs(1. - best(0)), 0.0001);
//     ASSERT_LE(std::abs(3. - best(1)), 0.0001);
// }