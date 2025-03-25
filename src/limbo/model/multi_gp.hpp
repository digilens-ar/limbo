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
#ifndef LIMBO_MODEL_MULTI_GP_HPP
#define LIMBO_MODEL_MULTI_GP_HPP

#include <limbo/mean/null_function.hpp>
#include <limbo/tools/parallel.hpp>

namespace limbo {
    namespace model {
        /// @ingroup model
        /// A wrapper for N-output Gaussian processes.
        /// It is parametrized by:
        /// - GaussianProcess class
        /// - a kernel function (the same type for all GPs, but can have different parameters)
        /// - a mean function (the same type and parameters for all GPs)
        /// - [optional] an optimizer for the hyper-parameters
        template <template <typename, typename, typename> class GPClass, typename KernelFunction, typename MeanFunction, class HyperParamsOptimizer = limbo::model::gp::NoLFOpt>
        class MultiGP {
        public:
            using GP_t = GPClass<KernelFunction, limbo::mean::NullFunction, limbo::model::gp::NoLFOpt>;

            /// useful because the model might be created before having samples
            MultiGP(int dim_in, int dim_out)
                : _dim_in(dim_in), _dim_out(dim_out), _mean_function()
            {
                // initialize dim_in models with 1 output
                for (int i = 0; i < _dim_out; i++) {
                    _gp_models.emplace_back(_dim_in, 1);
                }
            }

            /// Compute the GaussianProcess from samples and observations. This call needs to be explicit!
            static MultiGP createFromSamples(const std::vector<Eigen::VectorXd>& samples, const std::vector<Eigen::VectorXd>& observations)
            {
                assert(samples.size() != 0);
                assert(observations.size() != 0);
                assert(samples.size() == observations.size());
         
                MultiGP out(samples[0].size(), observations[0].size());

                // save observations
                // TO-DO: Check how can we improve for not saving observations twice (one here and one for each GaussianProcess)!?
                out._observations = observations;

                // compute the new observations for the GPs
                std::vector<std::vector<Eigen::VectorXd>> obs(out._dim_out);

                for (size_t j = 0; j < observations.size(); j++) {
                    Eigen::VectorXd mean_vector = _mean_function(samples[j], out);
                    assert(mean_vector.size() == out._dim_out);
                    for (int i = 0; i < out._dim_out; i++) {
                        obs[i].push_back(Eigen::VectorXd { {observations[j][i] - mean_vector[i]} });
                    }
                }

                // do the actual computation
                limbo::tools::par::loop(0, out._dim_out, [&](size_t i) {
                    out._gp_models[i] = GP_t::createFromSamples(samples, obs[i]);
                });
                return out;
            }

            /// Do not forget to call this if you use hyper-parameters optimization!!
            void optimize_hyperparams()
            {
                _hp_optimize(*this);
            }

            const MeanFunction& mean_function() const { return _mean_function; }

            /// add sample and update the GPs. This code uses an incremental implementation of the Cholesky
            /// decomposition. It is therefore much faster than a call to compute()
            void add_sample(const Eigen::VectorXd& sample, const Eigen::VectorXd& observation)
            {
                assert(sample.size() == _dim_in);
                assert(observation.size() == _dim_out);

                _observations.push_back(observation);

                Eigen::VectorXd mean_vector = _mean_function(sample, *this);
                assert(mean_vector.size() == _dim_out);

                limbo::tools::par::loop(0, _dim_out, [&](size_t i) {
                    _gp_models[i].add_sample(sample, { {observation[i] - mean_vector[i]} });
                });
            }

            /**
             \\rst
             return :math:`\mu`, :math:`\sigma^2` (un-normalized; this will return a vector --- one for each GaussianProcess). Using this method instead of separate calls to mu() and sigma() is more efficient because some computations are shared between mu() and sigma().
             \\endrst
            */
            std::tuple<Eigen::VectorXd, Eigen::VectorXd> query(const Eigen::VectorXd& v) const
            {
                Eigen::VectorXd mu(_dim_out);
                Eigen::VectorXd sigma(_dim_out);

                // query the mean function
                Eigen::VectorXd mean_vector = _mean_function(v, *this);

                // parallel query of the GPs
                limbo::tools::par::loop(0, _dim_out, [&](size_t i) {
                    Eigen::VectorXd tmp;
                    std::tie(tmp, sigma(i)) = _gp_models[i].query(v);
                    mu(i) = tmp(0) + mean_vector(i);
                });

                return std::make_tuple(mu, sigma);
            }

            /**
             \\rst
             return :math:`\mu` (un-normalized). If there is no sample, return the value according to the mean function.
             \\endrst
            */
            Eigen::VectorXd mu(const Eigen::VectorXd& v) const
            {
                Eigen::VectorXd mu(_dim_out);
                Eigen::VectorXd mean_vector = _mean_function(v, *this);

                limbo::tools::par::loop(0, _dim_out, [&](size_t i) {
                    mu(i) = _gp_models[i].mu(v)[0] + mean_vector(i);
                });

                return mu;
            }

            /**
             \\rst
             return :math:`\sigma^2` (un-normalized). This returns a vector; one value for each GaussianProcess.
             \\endrst
            */
            Eigen::VectorXd sigma(const Eigen::VectorXd& v) const
            {
                Eigen::VectorXd sigma(_dim_out);

                limbo::tools::par::loop(0, _dim_out, [&](size_t i) {
                    sigma(i) = _gp_models[i].sigma_sq(v);
                });

                return sigma;
            }

            /// return the number of dimensions of the input
            int dim_in() const
            {
                return _dim_in;
            }

            /// return the number of dimensions of the output
            int dim_out() const
            {
                return _dim_out;
            }

            /// return the list of samples
            const std::vector<Eigen::VectorXd>& samples() const
            {
                return _gp_models[0].samples();
            }

            /// return the list of observations
            const std::vector<Eigen::VectorXd>& observations() const
            {
                return _observations;
            }

            /// return the mean observation
            Eigen::VectorXd mean_observation() const
            {
                assert(_dim_out > 0);
                Eigen::VectorXd mean_observation = Eigen::VectorXd::Zero(_dim_out);
                for (size_t j = 0; j < _observations.size(); j++)
                    mean_observation.array() += _observations[j].array();
                mean_observation.array() /= static_cast<double>(_observations.size());
                return _observations.size() > 0 ? mean_observation : Eigen::VectorXd::Zero(_dim_out);
            }

            /// return the list of GPs
            std::vector<GP_t> const& gp_models() const
            {
                return _gp_models;
            }

            /// return the list of GPs
            std::vector<GP_t>& gp_models()
            {
                return _gp_models;
            }

            /// save the parameters and the data for the GaussianProcess to the archive (text or binary)
            template <typename A>
            void save(const A& archive) const
            {
                Eigen::VectorXd dims(2);
                dims << _dim_in, _dim_out;
                archive.save(dims, "dims");

                archive.save(_observations, "observations");

                if (_mean_function.h_params_size() > 0) {
                    archive.save(_mean_function.h_params(), "mean_params");
                }

                for (int i = 0; i < _dim_out; i++) {
                    _gp_models[i].save(A(archive.directory() + "/gp_" + std::to_string(i)));
                }
            }

            /// load the parameters and the data for the GaussianProcess from the archive (text or binary)
            /// if recompute_ is true, we do not read the kernel matrix
            /// but we recompute_ it given the data and the hyperparameters
            template <typename A>
            static MultiGP load(const A& archive, bool recompute = true)
            {

                Eigen::VectorXd dims;
                archive.load(dims, "dims");

                MultiGP out(dims(0), dims(1));
                out._observations.clear();
                archive.load(out._observations, "observations");

                out._mean_function = MeanFunction(out._dim_out);

                if (out._mean_function.h_params_size() > 0) {
                    Eigen::VectorXd h_params;
                    archive.load(h_params, "mean_params");
                    assert(h_params.size() == (int)out._mean_function.h_params_size());
                    out._mean_function.set_h_params(h_params);
                }

                out._gp_models.clear();

                for (int i = 0; i < out._dim_out; i++) {
                    // do not recompute_ the individual GPs on their own
                    out._gp_models.emplace_back(GP_t::load(A(archive.directory() + "/gp_" + std::to_string(i)), false));
                }

                if (recompute)
                    out.recompute_(true, true);

                return out;
            }

        private:
            std::vector<GP_t> _gp_models;
            int _dim_in, _dim_out;
            HyperParamsOptimizer _hp_optimize;
            MeanFunction _mean_function;
            std::vector<Eigen::VectorXd> _observations;

            ///  recomputes the GPs
            void recompute_(bool update_obs_mean, bool update_full_kernel)
            {
                // if there are no GPs, there's nothing to recompute_
                if (_gp_models.size() == 0)
                    return;

                if (update_obs_mean) // if the mean is updated, we need to fully re-compute
                    return initialize(_gp_models[0].samples(), _observations, update_full_kernel);
                else
                    limbo::tools::par::loop(0, _dim_out, [&](size_t i) {
                    _gp_models[i].recompute_(false, update_full_kernel);
                        });
            }
        };
    } // namespace model
} // namespace limbo

#endif