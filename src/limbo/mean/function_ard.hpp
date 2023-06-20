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
#ifndef LIMBO_MEAN_FUNCTION_ARD_HPP
#define LIMBO_MEAN_FUNCTION_ARD_HPP

#include <limbo/mean/mean.hpp>

namespace limbo {
    namespace mean {

        /// Functor used to optimize the mean function using the maximum likelihood principle
        ///
        /// It incorporates the hyperparameters of the underlying mean function, if any
        /// @see limbo::model::gp::KernelMeanLFOpt, limbo::model::gp::MeanLFOpt
        template <typename MeanFunction>
        struct FunctionARD : public BaseMean {
            FunctionARD()
                : _mean_function(), _tr(2)
            {
                Eigen::VectorXd h = Eigen::VectorXd::Zero(2 + _mean_function.h_params_size());
                h[0] = 1;
                if (_mean_function.h_params_size() > 0)
                    h.tail(_mean_function.h_params_size()) = _mean_function.h_params();
                this->set_h_params(h);
            }

            size_t h_params_size() const { return _tr.rows() + _mean_function.h_params_size(); }

            Eigen::VectorXd h_params() const
            {
                Eigen::VectorXd params(h_params_size());
                params.head(_tr.rows()) = _h_params;
                if (_mean_function.h_params_size() > 0)
                    params.tail(_mean_function.h_params_size()) = _mean_function.h_params();

                return params;
            }

            void set_h_params(const Eigen::VectorXd& p)
            {
                _h_params = p.head(_tr.rows());
                for (int c = 0; c < _tr.rows(); c++)
                        _tr(c) = p[c];

                if (_mean_function.h_params_size() > 0)
                    _mean_function.set_h_params(p.tail(_mean_function.h_params_size()));
            }

            template <typename GP>
            Eigen::VectorXd grad(const Eigen::VectorXd& x, const GP& gp) const
            {
                Eigen::VectorXd grad = Eigen::VectorXd::Zero(h_params_size());
                grad(0) = _mean_function(x, gp);
                grad(1) = 1;
                
                if (_mean_function.h_params_size() > 0) {
                    Eigen::MatrixXd m_grad = Eigen::MatrixXd::Zero(2, _mean_function.h_params_size());
                    m_grad.block(0, 0, 1, _mean_function.h_params_size()) = _mean_function.grad(x, gp).transpose();
                    Eigen::MatrixXd gg = _tr.transpose() * m_grad;
                    grad.tail(_mean_function.h_params_size()) = gg.transpose();
                }
                return grad;
            }

            template <typename GP>
            double operator()(const Eigen::VectorXd& x, const GP& gp) const
            {
                double m = _mean_function(x, gp);
                Eigen::VectorXd m_1(2);
                m_1(0) = m;
                m_1(1) = 1;
                return _tr.transpose() * m_1;
            }

        protected:
            MeanFunction _mean_function;
            Eigen::VectorXd _tr;
            Eigen::VectorXd _h_params;
        };
    } // namespace mean
} // namespace limbo

#endif
