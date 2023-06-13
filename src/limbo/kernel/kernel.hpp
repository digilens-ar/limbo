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
#ifndef LIMBO_KERNEL_KERNEL_HPP
#define LIMBO_KERNEL_KERNEL_HPP

#include <Eigen/Core>
#include <optional>
#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct kernel {
            /// @ingroup kernel_defaults
            BO_PARAM(double, noise, 0.01);
            BO_PARAM(bool, optimize_noise, false);
        };
    } // namespace defaults

    namespace kernel {
        /**
          @ingroup kernel
          \rst
          Base struct for kernel definition. It handles the noise and its optimization (only if the kernel allows hyper-parameters optimization).
          \endrst

          Parameters:
             - ``double noise`` (initial signal noise squared)
             - ``bool optimize_noise`` (whether we are optimizing for the noise or not)
        */
        template <typename kernel_opt, typename Kernel>
        struct BaseKernel {
        public:
            BaseKernel(size_t dim = 1) : _noise(kernel_opt::noise())
            {
                _noise_p = std::log(std::sqrt(_noise));
            }

            double compute(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2, int i = -1, int j = -2) const
            {
                double k = static_cast<const Kernel*>(this)->template kernel_<false>(v1, v2);
                return k + ((i == j) ? _noise + 1e-8 : 0.0);
            }

            Eigen::VectorXd grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, int i = -1, int j = -2) const
            {
                auto [k, grad] = static_cast<const Kernel*>(this)->template kernel_<true>(x1, x2);
                if (kernel_opt::optimize_noise()) {
                    grad.conservativeResize(grad.size() + 1);
                    grad(grad.size() - 1) = ((i == j) ? 2.0 * _noise : 0.0);
                }

                return grad;
            }

            // Get the hyper parameters size
            size_t h_params_size() const
            {
                return static_cast<const Kernel*>(this)->params_size() + (kernel_opt::optimize_noise() ? 1 : 0);
            }

            // Get the hyper parameters in log-space
            Eigen::VectorXd h_params() const
            {
                Eigen::VectorXd params = static_cast<const Kernel*>(this)->params();
                if (kernel_opt::optimize_noise()) {
                    params.conservativeResize(params.size() + 1);
                    params(params.size() - 1) = _noise_p;
                }
                return params;
            }

            // We expect the input parameters to be in log-space
            void set_h_params(const Eigen::VectorXd& p)
            {
                static_cast<Kernel*>(this)->set_params(p.head(h_params_size() - (kernel_opt::optimize_noise() ? 1 : 0)));
                if (kernel_opt::optimize_noise()) {
                    _noise_p = p(h_params_size() - 1);
                    _noise = std::exp(2 * _noise_p);
                }
            }

            // Get signal noise
            double noise() const { return _noise; }

        protected:
            double _noise;
            double _noise_p;
        };
    } // namespace kernel
} // namespace limbo

#endif
