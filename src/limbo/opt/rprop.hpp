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
#ifndef LIMBO_OPT_RPROP_HPP
#define LIMBO_OPT_RPROP_HPP

#include <algorithm>

#include <Eigen/Core>

#include <limbo/opt/optimizer.hpp>
#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct opt_rprop {
            /// @ingroup opt_defaults
            /// number of max iterations
            BO_PARAM(int, iterations, 300);

            /// gradient norm epsilon for stopping. Set this to some positive value for it to have an effect. Set to 0 to disable.
            BO_PARAM(double, eps_stop, 0);
        };
    }
    namespace opt {
        /// @ingroup opt
        /// Gradient-based optimization (rprop)
        /// - partly inspired by libgp: https://github.com/mblum/libgp
        /// - reference :
        /// Blum, M., & Riedmiller, M. (2013). Optimization of Gaussian
        /// Process Hyperparameters using Rprop. In European Symposium
        /// on Artificial Neural Networks, Computational Intelligence
        /// and Machine Learning.
        ///
        /// Parameters:
        /// - int iterations
        /// - double eps_stop
        template <typename opt_rprop>
        struct Rprop {
            static Rprop create(int dims)
            {
                return Rprop();
            }

            template <concepts::EvalFunc F>
            Eigen::VectorXd optimize(const F& f, const Eigen::VectorXd& init, std::optional<std::vector<std::pair<double, double>>> const& bounds) const
            {
                assert(opt_rprop::eps_stop() >= 0.);
                assert(!bounds.has_value() && "rprop doesn't suppoert bounds");

                const size_t param_dim = init.size();
                constexpr double delta0 = 0.1;
                constexpr double deltamin = 1e-6;
                constexpr double deltamax = 50;
                constexpr double etaminus = 0.5;
                constexpr double etaplus = 1.2;

                Eigen::VectorXd delta = Eigen::VectorXd::Ones(param_dim) * delta0;
                Eigen::VectorXd grad_old = Eigen::VectorXd::Zero(param_dim);
                Eigen::VectorXd params = init;

                Eigen::VectorXd best_params = params;
                double best = -INFINITY;

                for (int i = 0; i < opt_rprop::iterations(); ++i) {
                    auto [funcVal, gradient] = f(params, true);
                    if (funcVal > best) {
                        best = funcVal;
                        best_params = params;
                    }
                    Eigen::VectorXd grad = -gradient.value();
                    grad_old = grad_old.cwiseProduct(grad);

                    for (int j = 0; j < grad_old.size(); ++j) {
                        if (grad_old(j) > 0) {
                            delta(j) = std::min(delta(j) * etaplus, deltamax);
                        }
                        else if (grad_old(j) < 0) {
                            delta(j) = std::max(delta(j) * etaminus, deltamin);
                            grad(j) = 0;
                        }
                        params(j) += -signum(grad(j)) * delta(j);
                    }

                    grad_old = grad;
                    if (grad_old.norm() < opt_rprop::eps_stop())
                        break;
                }

                return best_params;
            }

        private:
            /// return -1 if x < 0;
            /// return 0 if x = 0;
            /// return 1 if x > 0.
            static int signum(double x)
            {
                return (0 < x) - (x < 0);
            }
        };
    }
}

#endif
