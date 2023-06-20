#pragma once

#include <algorithm>

#include <Eigen/Core>

#include <limbo/opt/optimizer.hpp>
#include <limbo/tools/macros.hpp>
#include <spdlog/spdlog.h>

namespace limbo {
    namespace defaults {
        struct opt_irpropplus {
            /// @ingroup opt_defaults
            /// number of max iterations
            BO_PARAM(int, max_iterations, 300);
            BO_PARAM(double, init_delta, 0.1);
            BO_PARAM(double, max_delta, 50);
            BO_PARAM(double, min_delta, 1e-6);

            /// gradient norm epsilon for stopping. Set this to some positive value for it to have an effect. Set to 0 to disable.
            BO_PARAM(double, min_gradient, 1e-10);
        };
    }
    namespace opt {
        /// @ingroup opt
        /// Gradient-based optimization (irprop_plus) An enhancement of rprop
        /// - partly inspired by: https://www.dropbox.com/s/ruytpw66g8097cb/contourplots.py?dl=0
        /// - reference: https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2003-Neuro-Igel-IRprop+.pdf
        /// - reference: https://citeseerx.ist.psu.edu/doc/10.1.1.17.1332
        ///
        /// Parameters:
        /// - int iterations
        /// - double eps_stop
        template <typename opt_irpropplus>
        struct Irpropplus {
            static Irpropplus create(int dims)
            {
                return Irpropplus();
            }

            template <concepts::EvalFunc F>
            Eigen::VectorXd optimize(const F& f, const Eigen::VectorXd& init, std::optional<std::vector<std::pair<double, double>>> const& bounds) const
            {

                const size_t param_dim = init.size();
                constexpr double delta0 = opt_irpropplus::init_delta();
                constexpr double deltamin = opt_irpropplus::min_delta();
                constexpr double deltamax = opt_irpropplus::max_delta();
                constexpr double etaminus = 0.5;
                constexpr double etaplus = 1.2;

                Eigen::VectorXd delta = Eigen::VectorXd::Constant(param_dim, delta0); // The magnitude of the amount to change the weigths on each iteration
                Eigen::VectorXd deltaWeights = delta; // The amount to change each weight. Store this to use for backtracking.
                Eigen::VectorXd grad_old = Eigen::VectorXd::Zero(param_dim); // The previous iteration's gradient
                Eigen::VectorXd weights = init; // The working copy of the weights

                Eigen::VectorXd best_params = weights;
                double best = INFINITY;
                double lastVal = best;

                int cnt = 0;
                for (int i = 0; i < opt_irpropplus::max_iterations(); ++i) {
                    auto [funcVal, gradient] = f(weights, true); // Evaluate the function at the current paramter values.
                    funcVal = -funcVal; // invert the value and gradient since we are maximizing but the original algorithm is written for minimizing.
                    Eigen::VectorXd grad = -gradient.value();

                	if (funcVal < best) {
                        best = funcVal;
                        best_params = weights;
                    }

                    if (grad.norm() < opt_irpropplus::min_gradient())
                    {
                        grad_old = grad;
                        break;
                    }

                    for (int j = 0; j < grad_old.size(); ++j) {
                        double gradProduct = grad(j) * grad_old(j);
                        if (gradProduct > 0) 
                        { // Gradient is continuing in same dimension
                            delta(j) = std::min(delta(j) * etaplus, deltamax);
                            deltaWeights(j) = -signum(grad(j)) * delta(j);
                            weights(j) += deltaWeights(j);
                        }
                        else if (gradProduct < 0) 
                        { // Gradient has changed sign
                            delta(j) = std::max(delta(j) * etaminus, deltamin);
                            if (funcVal > lastVal)
                            { // This iteration is worst than the last one
                                weights(j) -= deltaWeights(j);
                            }
                            grad(j) = 0;
                        }
                        else
                        { // The product is 0
                            deltaWeights(j) = -signum(grad(j)) * delta(j);
                            weights(j) += deltaWeights(j);
                        }

                        if (bounds.has_value())
                            weights(j) = std::clamp(weights(j), bounds.value().at(j).first, bounds.value().at(j).second);
                    }
                    lastVal = funcVal;
                    grad_old = grad;
                    ++cnt;
                }
                spdlog::info("Irpropplus completed in {} iterations. Gradient {}", cnt, grad_old.norm());
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
