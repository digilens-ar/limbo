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
#ifndef LIMBO_MODEL_GP_HPP
#define LIMBO_MODEL_GP_HPP

#include <cassert>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

// Quick hack for definition of 'I' in <complex.h>
#undef I

#include <numeric>
#include <limbo/kernel/matern_five_halves.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/data.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>
#include <limbo/model/gp/no_lf_opt.hpp>
#include <spdlog/spdlog.h>
#ifdef _WIN32
#include <corecrt_math_defines.h>
#endif

namespace limbo {
    namespace model {
        /// @ingroup model
        /// A classic Gaussian process.
        /// It is parametrized by:
        /// - a kernel function
        /// - a mean function
        /// - [optional] an optimizer for the hyper-parameters
        template <typename KernelFunction, std::derived_from<mean::BaseMean> MeanFunction = mean::Data, typename HyperParamsOptimizer = gp::NoLFOpt>
        class GaussianProcess {
        public:
            explicit GaussianProcess(int dim_in)
                : _dim_in(dim_in), _kernel_function(dim_in), _mean_function(), _inv_kernel_updated(false), _hp_optimize(HyperParamsOptimizer::create(dim_in)) {}

            /// Initialize the GaussianProcess from samples and observations. This call needs to be explicit!
            void initialize(std::vector<Eigen::VectorXd> samples, std::vector<double> observations)
            {
                assert(samples.size() != 0);
                assert(observations.size() != 0);
                assert(samples.size() == observations.size());
                assert(_dim_in == samples[0].size());

                _samples = std::move(samples);
                _observations = std::move(observations);

                //calculate the mean observation
                observation_sum_ = std::accumulate(_observations.begin(), _observations.end(), 0.0);
                observation_mean_ = observation_sum_ / _observations.size();

                this->_compute_observation_deviation();
            	this->_compute_full_kernel();
            }

            /// Do not forget to call this if you use hyper-parameters optimization!!
            void optimize_hyperparams()
            {
                _hp_optimize(*this);
            }

            /// add sample and update the GaussianProcess. This code uses an incremental implementation of the Cholesky
            /// decomposition. It is therefore much faster than a call to compute()
            void add_sample(const Eigen::VectorXd& sample, double observation)
            {
                assert(sample.size() == _dim_in);

                _samples.push_back(sample);
                _observations.push_back(observation);
                observation_sum_ += observation;
                observation_mean_ = observation_sum_ / _observations.size();

                this->_compute_observation_deviation();
                this->_compute_incremental_kernel();
            }

            /**
            
             return :math:`\mu`, :math:`\sigma^2` (un-normalized). If there is no sample, return the value according to the mean function. Using this method instead of separate calls to mu() and sigma() is more efficient because some computations are shared between mu() and sigma().
            */
            std::tuple<double, double> query(const Eigen::VectorXd& v) const
            {
                if (_samples.size() == 0)
                    return std::make_tuple(
                        _mean_function(v, *this),
                        _kernel_function.compute(v, v) + _kernel_function.noise());

                Eigen::VectorXd k = _compute_k(v);
                return std::make_tuple(_mu(v, k), _sigma_sq(v, k) + _kernel_function.noise());
            }

            /**
             \\rst
             return :math:`\mu` (un-normalized). If there is no sample, return the value according to the mean function.
             \\endrst
            */
            double mu(const Eigen::VectorXd& v) const
            {
                if (_samples.size() == 0)
                    return _mean_function(v, *this);
                return _mu(v, _compute_k(v));
            }

            /**
             \\rst
             return :math:`\sigma^2` (un-normalized). If there is no sample, return the max :math:`\sigma^2`.
             \\endrst
            */
            double sigma_sq(const Eigen::VectorXd& v) const
            {
                if (_samples.size() == 0)
                    return _kernel_function.compute(v, v) + _kernel_function.noise();
                return _sigma_sq(v, _compute_k(v)) + _kernel_function.noise();
            }

            /// return the number of dimensions of the input
            int dim_in() const
            {
                return _dim_in;
            }

            KernelFunction const& kernel_function() const { return _kernel_function; }

            MeanFunction const& mean_function() const { return _mean_function; }

            void set_kernel_hyperparams(Eigen::VectorXd const& hp)
            {
                _kernel_function.set_h_params(hp);
                recompute_(false, true);
            }

            void set_mean_hyperparams(Eigen::VectorXd const& hp)
            {
                _mean_function.set_h_params(hp);
                recompute_(true, false);
            }

            void set_all_hyperparams(Eigen::VectorXd const& kernel_hp, Eigen::VectorXd const& mean_hp)
            {
                _mean_function.set_h_params(mean_hp);
                _kernel_function.set_h_params(kernel_hp);
                recompute_(true, true);
            }

            /// return the mean observation
            double mean_observation() const
            {
                return observation_mean_;
            }

            /// compute and return the log likelihood
            [[nodiscard]] double compute_log_lik()
            {
                // --- cholesky ---
                // see:
                // http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
                long double logdet = 2 * _matrixL.diagonal().array().log().sum();

                const double a = observation_deviation_.transpose() * _alpha;
                const double log_2_pi = std::log(2 * M_PI);
                return 0.5 * (-a - logdet - observation_deviation_.rows() * log_2_pi);
            }

            /// compute and return the gradient of the log likelihood wrt to the kernel parameters
            [[nodiscard]] Eigen::VectorXd compute_kernel_grad_log_lik()
            {
                size_t n = observation_deviation_.rows();

                // compute K^{-1} only if needed
                if (!_inv_kernel_updated) {
                    compute_inv_kernel_();
                }

                // alpha * alpha.transpose() - K^{-1}
                Eigen::MatrixXd w = _alpha * _alpha.transpose() - _inv_kernel;

                // only compute half of the matrix (symmetrical matrix)
                Eigen::VectorXd grad = Eigen::VectorXd::Zero(_kernel_function.h_params_size());
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j <= i; ++j) {
                        Eigen::VectorXd g = _kernel_function.grad(_samples[i], _samples[j], i, j);
                        if (i == j)
                            grad += w(i, j) * g * 0.5;
                        else
                            grad += w(i, j) * g;
                    }
                }
                return grad;
            }

            /// compute and return the gradient of the log likelihood wrt to the mean parameters
            [[nodiscard]] Eigen::VectorXd compute_mean_grad_log_lik()
            {
                // compute K^{-1} only if needed
                if (!_inv_kernel_updated) {
                    compute_inv_kernel_();
                }

                Eigen::VectorXd grad = Eigen::VectorXd::Zero(_mean_function.h_params_size());
                for (Eigen::Index n_obs = 0; n_obs < observation_deviation_.rows(); n_obs++) {
                    grad += observation_deviation_.transpose() * _inv_kernel.col(n_obs) * _mean_function.grad(_samples[n_obs], *this).transpose();
                }

                return grad;
            }

            /// compute and return the log probability of LOO CV
            [[nodiscard]] double compute_log_loo_cv()
            {
                // compute K^{-1} only if needed
                if (!_inv_kernel_updated) {
                    compute_inv_kernel_();
                }

                Eigen::VectorXd inv_diag = _inv_kernel.diagonal().array().inverse();

                double log_loo_cv = (((-0.5 * (_alpha.array().square().array().colwise() * inv_diag.array())).array().colwise() - 0.5 * inv_diag.array().log().array()) - 0.5 * std::log(2 * M_PI)).colwise().sum().sum();
                return log_loo_cv;
            }

            /// compute and return the gradient of the log probability of LOO CV w.r.t. to the kernel parameters
            [[nodiscard]] Eigen::VectorXd compute_kernel_grad_log_loo_cv()
            {
                size_t n = observation_deviation_.rows();
                size_t n_params = _kernel_function.h_params_size();

                // compute K^{-1} only if needed
                if (!_inv_kernel_updated) {
                    compute_inv_kernel_();
                }

                Eigen::VectorXd grad = Eigen::VectorXd::Zero(n_params);
                Eigen::MatrixXd grads = Eigen::MatrixXd::Zero(n_params, 1);

                // only compute half of the matrix (symmetrical matrix)
                // TO-DO: Make it better
                std::vector<std::vector<Eigen::VectorXd>> full_dk;
                for (size_t i = 0; i < n; i++) {
                    full_dk.push_back(std::vector<Eigen::VectorXd>());
                    for (size_t j = 0; j <= i; j++)
                        full_dk[i].push_back(_kernel_function.grad(_samples[i], _samples[j], i, j));
                    for (size_t j = i + 1; j < n; j++)
                        full_dk[i].push_back(Eigen::VectorXd::Zero(n_params));
                }
                for (size_t i = 0; i < n; i++)
                    for (size_t j = 0; j < i; ++j)
                        full_dk[j][i] = full_dk[i][j];

                Eigen::VectorXd inv_diag = _inv_kernel.diagonal().array().inverse();

                for (int j = 0; j < grad.size(); j++) {
                    Eigen::MatrixXd dKdTheta_j = Eigen::MatrixXd::Zero(n, n);
                    for (size_t i = 0; i < n; i++) {
                        for (size_t k = 0; k < n; k++)
                            dKdTheta_j(i, k) = full_dk[i][k](j);
                    }
                    Eigen::MatrixXd Zeta_j = _inv_kernel * dKdTheta_j;
                    Eigen::MatrixXd Zeta_j_alpha = Zeta_j * _alpha;
                    Eigen::MatrixXd Zeta_j_K = Zeta_j * _inv_kernel;

                    grads.row(j) = ((_alpha.array() * Zeta_j_alpha.array() - 0.5 * ((1. + _alpha.array().square().array().colwise() * inv_diag.array()).array().colwise() * Zeta_j_K.diagonal().array())).array().colwise() * inv_diag.array()).colwise().sum();
                }

                grad = grads.rowwise().sum();

                return grad;
            }

            /// return the list of samples
            std::vector<Eigen::VectorXd> const& samples() const { return _samples; }

            /// return the list of observations
            std::vector<double> const& observations() const { return _observations; }

            std::pair<double, Eigen::VectorXd> best_observation() const
            {
                const auto max_e = std::max_element(observations().begin(), observations().end());
                const auto index = std::distance(observations().begin(), max_e);
                return { observations()[index], samples()[index] };
            }

            /// save the parameters and the data for the GaussianProcess to the archive (text or binary)
            template <concepts::Archive A>
            void save(A const& archive) const
            {
                if (_kernel_function.h_params_size() > 0) {
                    archive.save(_kernel_function.h_params(), "kernel_params");
                }
                if (_mean_function.h_params_size() > 0) {
                    archive.save(_mean_function.h_params(), "mean_params");
                }
                archive.save(_samples, "samples");
                archive.save(_observations, "observations");
                archive.save(_matrixL, "matrixL");
                archive.save(_alpha, "alpha");
            }

            /// load the parameters and the data for the GaussianProcess from the archive (text or binary)
            /// if recompute_ is true, we do not read the kernel matrix
            /// but we recompute_ it given the data and the hyperparameters
            template <concepts::Archive A>
            static GaussianProcess load(A const& archive, bool recompute = true)
            {
                std::vector<Eigen::VectorXd> samples;
                archive.load(samples, "samples");

                std::vector<double> observations;
                archive.load(observations, "observations");

                GaussianProcess out(samples[0].size());
                out._samples = samples;
                out._observations = observations;

                //calcualte the mean observation
                out.observation_sum_ = std::accumulate(out._observations.begin(), out._observations.end(), 0.0);
                out.observation_mean_ = out.observation_sum_ / out._observations.size();

                if (out._kernel_function.h_params_size() > 0) {
                    Eigen::VectorXd h_params;
                    archive.load(h_params, "kernel_params");
                    assert(h_params.size() == (int)out._kernel_function.h_params_size());
                    out._kernel_function.set_h_params(h_params);
                }

                out._mean_function = MeanFunction();

                if (out._mean_function.h_params_size() > 0) {
                    Eigen::VectorXd h_params;
                    archive.load(h_params, "mean_params");
                    assert(h_params.size() == (int)out._mean_function.h_params_size());
                    out._mean_function.set_h_params(h_params);
                }

                if (recompute)
                    out.recompute_(true, true);
                else {
                    archive.load(out._matrixL, "matrixL");
                    archive.load(out._alpha, "alpha");
                }
                return out;
            }

        private:
            int _dim_in;

            KernelFunction _kernel_function;
            MeanFunction _mean_function;

            std::vector<Eigen::VectorXd> _samples;
            std::vector<double> _observations;
            Eigen::VectorXd observation_deviation_; // The differences between the observations at the mean function at the observation locations
            double observation_sum_ = 0;
            double observation_mean_ = 0;

            Eigen::VectorXd _alpha;  // alpha = K^{-1} * this->observation_deviation_;
            Eigen::MatrixXd _kernel;
        	Eigen::MatrixXd _inv_kernel;

            Eigen::MatrixXd _matrixL;     /// The L matrix from LLT Cholesky decomposition

        	bool _inv_kernel_updated;

            HyperParamsOptimizer _hp_optimize;

            void _compute_observation_deviation()
            {
                assert(!_samples.empty());
                observation_deviation_.resize(_samples.size());
                for (int i = 0; i < observation_deviation_.rows(); i++) {
                    assert(_samples[i].cols() == 1);
                    assert(_samples[i].rows() != 0);
                    assert(_samples[i].rows() == _dim_in);
                    observation_deviation_(i) = _observations.at(i) - _mean_function(_samples[i], *this);
                }
            }

            void _compute_full_kernel()
            {
                size_t n = _samples.size();
                _kernel.resize(n, n);

                // Compute lower triangle
                for (size_t i = 0; i < n; i++)
                    for (size_t j = 0; j <= i; ++j)
                        _kernel(i, j) = _kernel_function.compute(_samples[i], _samples[j], i, j);

                // Copy lower triangle to top (TODO is this needed?)
                for (size_t i = 0; i < n; i++)
                    for (size_t j = 0; j < i; ++j)
                        _kernel(j, i) = _kernel(i, j);

                // O(n^3)
                _matrixL = Eigen::LLT<Eigen::MatrixXd>(_kernel).matrixL(); // _matrixL * _matrixL.transpose = _kernel
                this->_compute_alpha();

                // notify change of kernel
                _inv_kernel_updated = false;
            }

            void _compute_incremental_kernel()
            {
                // Incremental LLT
                // This part of the code is inspired from the Bayesopt Library (cholesky_add_row function).
                // However, the mathematical foundations can be easily retrieved by detailing the equations of the
                // extended L matrix that produces the desired kernel.

                size_t n = _samples.size();
                _kernel.conservativeResize(n, n);

                for (size_t i = 0; i < n; ++i) {
                    _kernel(i, n - 1) = _kernel_function.compute(_samples[i], _samples[n - 1], i, n - 1);
                    _kernel(n - 1, i) = _kernel(i, n - 1);
                }

                _matrixL.conservativeResizeLike(Eigen::MatrixXd::Zero(n, n));

                double L_j;
                for (size_t j = 0; j < n - 1; ++j) {
                    L_j = _kernel(n - 1, j) - (_matrixL.block(j, 0, 1, j) * _matrixL.block(n - 1, 0, 1, j).transpose())(0, 0);
                    _matrixL(n - 1, j) = (L_j) / _matrixL(j, j);
                }

                L_j = _kernel(n - 1, n - 1) - (_matrixL.block(n - 1, 0, 1, n - 1) * _matrixL.block(n - 1, 0, 1, n - 1).transpose())(0, 0);
                _matrixL(n - 1, n - 1) = sqrt(L_j);

                this->_compute_alpha();

                // notify change of kernel
                _inv_kernel_updated = false;
            }

            void _compute_alpha()
            {
                // alpha = K^{-1} * this->observation_deviation_;
                Eigen::TriangularView<Eigen::MatrixXd, Eigen::Lower> triang = _matrixL.triangularView<Eigen::Lower>();
                _alpha = triang.solve(observation_deviation_);
                triang.adjoint().solveInPlace(_alpha);
            }

            double _mu(Eigen::VectorXd const& v, Eigen::VectorXd const& k) const
            {
                return (k.transpose() * _alpha) + _mean_function(v, *this);
            }

            double _sigma_sq(Eigen::VectorXd const& v, Eigen::VectorXd const& k) const
            {
                Eigen::VectorXd z = _matrixL.triangularView<Eigen::Lower>().solve(k);
                double res = _kernel_function.compute(v, v) - z.dot(z);

                return (res <= std::numeric_limits<double>::epsilon()) ? 0 : res;
            }

            Eigen::VectorXd _compute_k(Eigen::VectorXd const& v) const
            {
                Eigen::VectorXd k(_samples.size());
                for (int i = 0; i < k.size(); i++)
                    k[i] = _kernel_function.compute(_samples[i], v);
                return k;
            }

            ///  recomputes the GaussianProcess
            void recompute_(bool update_obs_mean, bool update_full_kernel)
            {
                assert(!_samples.empty());

                if (update_obs_mean)
                    this->_compute_observation_deviation();

                if (update_full_kernel)
                    this->_compute_full_kernel();
                else
                    this->_compute_alpha();
            }

            void compute_inv_kernel_()
            {
                const size_t n = observation_deviation_.rows();
                // K^{-1} using Cholesky decomposition
                _inv_kernel = Eigen::MatrixXd::Identity(n, n);

                _matrixL.triangularView<Eigen::Lower>().solveInPlace(_inv_kernel);
                _matrixL.triangularView<Eigen::Lower>().transpose().solveInPlace(_inv_kernel);

                _inv_kernel_updated = true;
            }
        };

        /// GPBasic is a GaussianProcess with a "mean data" mean function, Exponential kernel,
        /// and NO hyper-parameter optimization
        template <typename Params>
        using GPBasic = GaussianProcess<kernel::MaternFiveHalves<typename Params::kernel, typename Params::kernel_maternfivehalves>, mean::Data, gp::NoLFOpt>;

        /// GPOpt is a GaussianProcess with a "mean data" mean function, Exponential kernel with Automatic Relevance
        /// Determination (ARD), and hyper-parameter optimization based on Rprop
        template <typename Params>
        using GPOpt = GaussianProcess<kernel::SquaredExpARD<typename Params::kernel, typename Params::kernel_squared_exp_ard>, mean::Data, gp::KernelLFOpt<opt::Irpropplus<typename Params::opt_irpropplus>>>;
    } // namespace model
} // namespace limbo

#endif
