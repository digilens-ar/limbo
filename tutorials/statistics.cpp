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
// please see the explanation in the documentation

#include <iostream>
#include <limbo/limbo.hpp>

#include "limbo/bayes_opt/boptimizer.hpp"

using namespace limbo;

struct Params {
    struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
        BO_PARAM(int, stats_enabled, true);
    };

// depending on which internal optimizer we use, we need to import different parameters
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

    // no noise
    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 1e-10);
    };

    struct kernel_maternfivehalves : public defaults::kernel_maternfivehalves {
    };

    // we use 10 random samples to initialize the algorithm
    struct init_randomsampling {
        BO_PARAM(int, samples, 10);
    };

    // we stop after 40 iterations
    struct stop_maxiterations : defaults::stop_maxiterations{
        BO_PARAM(int, iterations, 40);
    };

    // we use the default parameters for acqui_ucb
    struct acqui_ucb : public defaults::acqui_ucb {
    };
};

struct Eval {
    // number of input dimension (x.size())
    BO_PARAM(size_t, dim_in, 1);

    // the function to be optimized
    std::tuple<EvaluationStatus, double> operator()(const Eigen::VectorXd& x) const
    {
        double y = -((5 * x(0) - 2.5) * (5 * x(0) - 2.5)) + 5;
        // we return a 1-dimensional vector
        return { OK, y };
    }
};

template <typename Params>
struct WorstObservation : public limbo::stat::StatBase {
    template <typename BO>
    void operator()(const BO& bo)
    {
        // [optional] if statistics have been disabled or if there are no observations, we do not do anything
        if (bo.model().observations().empty())
            return;

        // [optional] we create a file to write / you can use your own file but remember that this method is called at each iteration (you need to create it in the constructor)
        auto& logFile = get_log_file("worst_observations.dat");

        // [optional] we add a header to the file to make it easier to read later
        if (bo.total_iterations() == 0)
            logFile << "#iteration worst_observation sample" << std::endl;

        // ----- search for the worst observation ----
        // 1. get the aggregated observations
        auto rewards = bo.model().observations();
        // 2. search for the worst element
        auto min_e = std::min_element(rewards.begin(), rewards.end());
        auto min_obs = bo.model().observations()[std::distance(rewards.begin(), min_e)];
        auto min_sample = bo.model().samples()[std::distance(rewards.begin(), min_e)];

        // ----- write what we have found ------
        logFile << bo.total_iterations() << " " << min_obs << " " << min_sample.transpose() << std::endl;
    }
};

int main()
{
    // we use the default acquisition function / model / stat / etc.

    // define a special list of statistics which include our new statistics class
    using stat_t = boost::fusion::vector<limbo::stat::ConsoleSummary,
        limbo::stat::Samples,
        limbo::stat::Observations,
        WorstObservation<Params>>;

    /// remmeber to use the new statistics vector via statsfun<>!
    using BD = bayes_opt::BOptimizer<Params>; // Default
    bayes_opt::BOptimizer<Params, BD::model_t, BD::acquisition_function_t, BD::init_function_t, BD::stopping_criteria_t, stat_t> boptimizer(1);

    // run the evaluation
    boptimizer.optimize(Eval());

    // the best sample found
    auto [bObs, bSamp] = boptimizer.model().best_observation();
    std::cout << "Best sample: " << bSamp(0) << " - Best observation: " << bObs << std::endl;
    return 0;
}
