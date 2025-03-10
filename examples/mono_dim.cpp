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
#include <limbo/acqui/gp_ucb.hpp>
#include <limbo/bayes_opt/boptimizer.hpp>
#include <limbo/kernel/kernel.hpp>
#include <limbo/mean/data.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/tools/macros.hpp>
#include "limbo/kernel/exp.hpp"
#include "limbo/opt/rprop.hpp"
#include <limbo/stat.hpp>
#include <boost/fusion/container.hpp>

using namespace limbo;

          struct Params {
              struct acqui_gpucb : public defaults::acqui_gpucb {
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
              struct acqui_ucb {
                  BO_PARAM(double, kappa, 0.1);
              };

              struct kernel : public defaults::kernel {
                  BO_PARAM(double, noise, 0.001);
              };

              struct kernel_maternfivehalves {
                  BO_PARAM(double, sigma_sq, 1);
                  BO_PARAM(double, l, 0.2);
              };
              struct kernel_exp : public defaults::kernel_exp {
              };

              struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
                  BO_PARAM(bool, stats_enabled, true);
              };

              struct init_randomsampling {
                  BO_PARAM(int, samples, 5);
              };

              struct stop_maxiterations : defaults::stop_maxiterations {
                  BO_PARAM(int, iterations, 20);
              };
              struct stat_gp {
                  BO_PARAM(int, bins, 20);
              };

              struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
              };

              struct opt_irpropplus : public defaults::opt_irpropplus {
              };
          };

struct fit_eval {
    BO_PARAM(size_t, dim_in, 2);

    std::tuple<EvaluationStatus, double> operator()(const Eigen::VectorXd& x) const
    {
        double res = 0;
        for (int i = 0; i < x.size(); i++)
            res += 1 - (x[i] - 0.3) * (x[i] - 0.3) + sin(10 * x[i]) * 0.2;
        return { OK, res };
    }
};

int main()
{
    using Kernel_t = kernel::MaternFiveHalves<Params::kernel, Params::kernel_maternfivehalves>;
    using Mean_t = mean::Data;
    using GP_t = model::GaussianProcess<Kernel_t, Mean_t>;
    using Acqui_t = acqui::UCB<Params::acqui_ucb, GP_t>;
    using stat_t = boost::fusion::vector<limbo::stat::ConsoleSummary,
        limbo::stat::Samples,
        limbo::stat::Observations,
        limbo::stat::GP<Params::stat_gp>>;

    using BD = bayes_opt::BOptimizer<Params>;
    bayes_opt::BOptimizer<Params, GP_t, Acqui_t, BD::init_function_t, BD::stopping_criteria_t, stat_t> opt(2);
    opt.optimize(fit_eval());
    auto [bestObs, bestSample] = opt.model().best_observation();
    std::cout << bestObs << " res  "
              << bestSample.transpose() << std::endl;

    // example with basic HP opt
    bayes_opt::BOptimizerHPOpt<Params> opt_hp(2);
    opt_hp.optimize(fit_eval());
    std::tie(bestObs, bestSample) = opt_hp.model().best_observation();

    std::cout << bestObs << " res  "
              << bestSample.transpose() << std::endl;
    return 0;
}
