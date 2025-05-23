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
#ifndef LIMBO_KERNEL_SQUARED_EXP_ARD_HPP
#define LIMBO_KERNEL_SQUARED_EXP_ARD_HPP

#include <limbo/kernel/kernel.hpp>

namespace limbo {
    namespace defaults {
        struct kernel_squared_exp_ard {
            /// @ingroup kernel_defaults
            BO_PARAM(int, k, 0); //equivalent to the standard exp ARD
            /// @ingroup kernel_defaults
            BO_PARAM(double, sigma_sq, 1);
  
        };
    } // namespace defaults

    namespace kernel {
        /**
        @ingroup kernel
        \rst

        Squared exponential covariance function with automatic relevance detection (to be used with a likelihood optimizer)
        Computes the squared exponential covariance like this:

        .. math::
            k_{SE}(v1, v2) = \sigma^2 \exp \Big(-\frac{1}{2}(v1-v2)^TM(v1-v2)\Big),

        with :math:`M = \Lambda\Lambda^T + diag(l_1^{-2}, \dots, l_n^{-2})` being the characteristic length scales and :math:`\alpha` describing the variability of the latent function. The parameters :math:`l_1^2, \dots, l_n^2, \Lambda,\sigma^2` are expected in this order in the parameter array. :math:`\Lambda` is a :math:`D\times k` matrix with :math:`k<D`.

        Parameters:
           - ``double sigma_sq`` (initial signal variance)
           - ``int k`` (number of columns of :math:`\Lambda` matrix)

        Reference: :cite:`Rasmussen2006`, p. 106 & :cite:`brochu2010tutorial`, p. 10
        \endrst
        */
        template <typename kernel_opt, typename kernel_squared_exp_ard>
        struct SquaredExpARD : BaseKernel<kernel_opt, SquaredExpARD<kernel_opt, kernel_squared_exp_ard>> {
            SquaredExpARD(int dim = 1) : _ell_inv(dim), _A(dim, kernel_squared_exp_ard::k()), _input_dim(dim)
            {
                Eigen::VectorXd p = Eigen::VectorXd::Zero(_ell_inv.size() + _ell_inv.size() * kernel_squared_exp_ard::k() + 1);
                p(p.size() - 1) = std::log(std::sqrt(kernel_squared_exp_ard::sigma_sq()));
                this->set_params_(p);
            }

        protected:
            Eigen::VectorXd gradient_(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
            {
                if (kernel_squared_exp_ard::k() > 0) {
                    Eigen::VectorXd grad = Eigen::VectorXd::Zero(this->params_size_());
                    Eigen::MatrixXd K = (_A * _A.transpose());
                    K.diagonal() += _ell_inv.array().square().matrix();
                    double z = ((x1 - x2).transpose() * K * (x1 - x2));
                    double k = _sf2 * std::exp(-0.5 * z);

                    grad.head(_input_dim) = (x1 - x2).cwiseProduct(_ell_inv).array().square() * k;

                    for (size_t j = 0; j < (unsigned int)kernel_squared_exp_ard::k(); ++j) {
                        Eigen::VectorXd G = -((x1 - x2).transpose() * _A.col(j))(0) * (x1 - x2) * k;
                        grad.segment((j + 1) * _input_dim, _input_dim) = G;
                    }

                    grad(grad.size() - 1) = 2 * k;

                    return grad;
                }
                else {
                    Eigen::VectorXd grad(this->params_size_());
                    Eigen::VectorXd z = (x1 - x2).cwiseProduct(_ell_inv).array().square();
                    const double k = _sf2 * std::exp(-0.5 * z.sum());
                    grad.head(_input_dim).noalias() = z * k;

                    grad(grad.size() - 1) = 2 * k;
                    return grad;
                }
            }

            double kernel_(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
            {
                double z;
                if (kernel_squared_exp_ard::k() > 0) {
                    Eigen::MatrixXd K = (_A * _A.transpose());
                    K.diagonal() += _ell_inv.array().square().matrix();
                    z = ((x1 - x2).transpose() * K * (x1 - x2));
                }
                else {
                    z = (x1 - x2).cwiseProduct(_ell_inv).squaredNorm();
                }
                return _sf2 * std::exp(-0.5 * z);
            }

            size_t params_size_() const { return _ell_inv.size() + _ell_inv.size() * kernel_squared_exp_ard::k() + 1; }

            // Return the hyper parameters in log-space
            Eigen::VectorXd params_() const { return _h_params; }

            // We expect the input parameters to be in log-space
            void set_params_(const Eigen::VectorXd& p)
            {
                _h_params = p;
                for (size_t i = 0; i < _input_dim; ++i)
                    _ell_inv(i) = 1.0 / std::exp(p(i));
                for (size_t j = 0; j < (unsigned int)kernel_squared_exp_ard::k(); ++j)
                    for (size_t i = 0; i < _input_dim; ++i)
                        _A(i, j) = p((j + 1) * _input_dim + i);
                _sf2 = std::exp(2.0 * p(params_size_() - 1));
            }

            double _sf2; // Sigma squared
            Eigen::VectorXd _ell_inv; // vector of 1 / l where l is the length scale for each axis.
            Eigen::MatrixXd _A;
            size_t _input_dim;
            Eigen::VectorXd _h_params; // The natural log of the length scales and sigma^2

            friend struct BaseKernel<kernel_opt, SquaredExpARD<kernel_opt, kernel_squared_exp_ard>>;
        };
    } // namespace kernel
} // namespace limbo

#endif
