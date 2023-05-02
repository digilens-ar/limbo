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
#ifndef LIMBO_BAYES_OPT_BOPTIMIZER_HPP
#define LIMBO_BAYES_OPT_BOPTIMIZER_HPP

#include <algorithm>
#include <iostream>
#include <iterator>
#include <filesystem>
#include <Eigen/Core>
#include <boost/fusion/container.hpp>
#include <boost/fusion/algorithm.hpp>

#include <limbo/tools/macros.hpp>
#include <limbo/tools/random_generator.hpp>
#include <limbo/stat.hpp>

#include "limbo/acqui/ucb.hpp"
#include "limbo/init/random_sampling.hpp"
#include "limbo/model/gp.hpp"
#include "limbo/stop/chain_criteria.hpp"
#include "limbo/stop/max_iterations.hpp"
#ifdef USE_NLOPT
#include <limbo/opt/nlopt_no_grad.hpp>
#elif defined USE_LIBCMAES
#include <limbo/opt/cmaes.hpp>
#else
#include <limbo/opt/grid_search.hpp>
#endif

namespace limbo {
    namespace defaults {
        struct bayes_opt_boptimizer {
            BO_PARAM(int, hp_period, -1); // If this is a positive number the model will `optimize_hyperparameters` every `hp_period` iterations
            BO_PARAM(bool, stats_enabled, true);
            BO_PARAM(bool, bounded, true);
        };
    }

    template <typename BO, typename AggregatorFunction>
    struct RefreshStat_f {
        RefreshStat_f(BO& bo, const AggregatorFunction& afun)
            : _bo(bo), _afun(afun) {}

        BO& _bo;
        const AggregatorFunction& _afun;

        template <typename T>
        void operator()(T& x) const { x(_bo, _afun); }
    };

    struct FirstElem {
        using result_type = double;
        double operator()(const Eigen::VectorXd& x) const
        {
            return x(0);
        }
    };
    class EvaluationError : public std::exception {};

    namespace bayes_opt {

              // defaults
			template<typename Params>
            struct defaults {
#ifdef USE_NLOPT
                using acquiopt_t = opt::NLOptNoGrad<typename Params::opt_nloptnograd, nlopt::GN_DIRECT_L_RAND>;
#elif defined(USE_LIBCMAES)
                using acquiopt_t = opt::Cmaes<Params>;
#else
#warning NO NLOpt, and NO Libcmaes: the acquisition function will be optimized by a grid search algorithm (which is usually bad). Please install at least NLOpt or libcmaes to use limbo!.
                using acquiopt_t = opt::GridSearch<Params>;
#endif
            };


        /**
       \rst

        The classic Bayesian optimization algorithm.


        \rst
        References: :cite:`brochu2010tutorial,Mockus2013`
        \endrst
        
       Parameters:
         - ``bool Params::bayes_opt_boptimizer::stats_enabled``: activate / deactivate the statistics

       This class is templated by several types with default values (thanks to boost::parameters).

		+----------------+---------+---------+---------------+
		|type            |typedef  | argument| default       |
		+================+=========+=========+===============+
		|init. func.     |init_t   | initfun | RandomSampling|
		+----------------+---------+---------+---------------+
		|model           |model_t  | modelfun| GP<...>       |
		+----------------+---------+---------+---------------+
		|acquisition fun.|aqui_t   | acquifun| GP_UCB        |
		+----------------+---------+---------+---------------+
		|statistics      | stat_t  | statfun | see below     |
		+----------------+---------+---------+---------------+
		|stopping crit.  | stop_t  | stopcrit| MaxIterations |
		+----------------+---------+---------+---------------+
		|acqui. optimizer|acquiopt_t| acquiopt | see below |
        +----------------+------------+----------+---------------+
       \endrst

       For GP, the default value is: ``model::GP<Params, kf_t, mean_t, opt_t>>``,
         - with ``kf_t = kernel::SquaredExpARD<Params>``
         - with ``mean_t = mean::Data<Params>``
         - with ``opt_t = model::gp::KernelLFOpt<Params>``

        (meaning: kernel with automatic relevance determination and mean equals to the mean of the input data, that is, center the data automatically)

       The default value of acqui_opt_t is:
        - ``opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND>`` if NLOpt was found in `waf configure`
        - ``opt::Cmaes<Params>`` if libcmaes was found but NLOpt was not found
        - ``opt::GridSearch<Params>`` otherwise (please do not use this: the algorithm will not work as expected!)


       For Statistics, the default value is: ``boost::fusion::vector<stat::Samples<Params>, stat::AggregatedObservations<Params>, stat::ConsoleSummary<Params>>``

       Example of customization:
         - ``using Kernel_t = kernel::MaternFiveHalves<Params>;``
         - ``using Mean_t = mean::Data<Params>;``
         - ``using GP_t = model::GP<Params, Kernel_t, Mean_t>;``
         - ``using Acqui_t = acqui::UCB<Params, GP_t>;``
         - ``bayes_opt::BOptimizer<Params, modelfun<GP_t>, acquifun<Acqui_t>> opt;``

       */
		template <
            class Params,
        	concepts::Model model_type = model::GP<kernel::MaternFiveHalves<typename Params::kernel, typename Params::kernel_maternfivehalves>>,
			typename acqui_t = acqui::UCB<typename Params::acqui_ucb, model_type>,
			typename init_t = init::RandomSampling<typename Params::init_randomsampling>,
    		typename StoppingCriteria = boost::fusion::vector<stop::MaxIterations<typename Params::stop_maxiterations>>,
    		typename Stat =  boost::fusion::vector<stat::Samples, stat::AggregatedObservations, stat::ConsoleSummary>,
			concepts::Optimizer acqui_opt_t = typename defaults<Params>::acquiopt_t
    	>
        class BOptimizer {
        public:
            // Public types
            using params_t = Params;
            using init_function_t = init_t;
            using acquisition_function_t = acqui_t;
            using model_t = model_type;
            using acqui_optimizer_t = acqui_opt_t;
            using stopping_criteria_t = typename boost::mpl::if_<boost::fusion::traits::is_sequence<StoppingCriteria>, StoppingCriteria, boost::fusion::vector<StoppingCriteria>>::type;
            using stat_t = typename boost::mpl::if_<boost::fusion::traits::is_sequence<Stat>, Stat, boost::fusion::vector<Stat>>::type;

            /// default constructor
            BOptimizer() : _total_iterations(0) { _make_res_dir(); }

            /// copy is disabled (dangerous and useless)
            BOptimizer(const BOptimizer& other) = delete;
            /// copy is disabled (dangerous and useless)
            BOptimizer& operator=(const BOptimizer& other) = delete;


            /// The main function (run the Bayesian optimization algorithm)
            template <concepts::StateFunc StateFunction, concepts::AggregatorFunc AggregatorFunction = FirstElem>
            void optimize(const StateFunction& sfun, const AggregatorFunction& afun = AggregatorFunction(), bool reset = true)
            {
                this->_init(sfun, afun, reset);

                if (!this->_observations.empty())
                    _model.compute(this->_samples, this->_observations);
                else
                    _model = model_t(sfun.dim_in(), sfun.dim_out());

                acqui_optimizer_t acqui_optimizer = acqui_optimizer_t::create(sfun.dim_in());

                while (!this->_stop(*this, afun)) {
                    acquisition_function_t acqui(_model, this->_current_iteration);

                    auto acqui_optimization = [&](const Eigen::VectorXd& x, bool g) -> opt::eval_t { return acqui(x, afun, g); };
                    Eigen::VectorXd starting_point = tools::random_vector(sfun.dim_in(), Params::bayes_opt_boptimizer::bounded());
                    Eigen::VectorXd new_sample = acqui_optimizer.optimize(acqui_optimization, starting_point, Params::bayes_opt_boptimizer::bounded());
                    this->eval_and_add(sfun, new_sample);

                    this->_update_stats(*this, afun);

                    _model.add_sample(this->_samples.back(), this->_observations.back());

                    if (Params::bayes_opt_boptimizer::hp_period() > 0
                        && (this->_current_iteration + 1) % Params::bayes_opt_boptimizer::hp_period() == 0)
                        _model.optimize_hyperparams();

                    this->_current_iteration++;
                    this->_total_iterations++;
                }
            }

            /// return the best observation so far (i.e. max(f(x)))
            template <concepts::AggregatorFunc AggregatorFunction = FirstElem>
            const Eigen::VectorXd& best_observation(const AggregatorFunction& afun = AggregatorFunction()) const
            {
                auto rewards = std::vector<double>(this->_observations.size());
                std::transform(this->_observations.begin(), this->_observations.end(), rewards.begin(), afun);
                auto max_e = std::max_element(rewards.begin(), rewards.end());
                return this->_observations[std::distance(rewards.begin(), max_e)];
            }

            /// return the best sample so far (i.e. the argmax(f(x)))
            template <concepts::AggregatorFunc AggregatorFunction = FirstElem>
            const Eigen::VectorXd& best_sample(const AggregatorFunction& afun = AggregatorFunction()) const
            {
                auto rewards = std::vector<double>(this->_observations.size());
                std::transform(this->_observations.begin(), this->_observations.end(), rewards.begin(), afun);
                auto max_e = std::max_element(rewards.begin(), rewards.end());
                return this->_samples[std::distance(rewards.begin(), max_e)];
            }

            const model_t& model() const { return _model; }

            /// return true if the statitics are enabled (they can be disabled to avoid dumping data, e.g. for unit tests)
            bool stats_enabled() const { return Params::bayes_opt_boptimizer::stats_enabled(); }

            /// return the name of the directory in which results (statistics) are written
            const std::string& res_dir() const { return _res_dir; }

            /// return the vector of points of observations (observations can be multi-dimensional, hence the VectorXd) -- f(x)
            const std::vector<Eigen::VectorXd>& observations() const { return _observations; }

            /// return the list of the points that have been evaluated so far (x)
            const std::vector<Eigen::VectorXd>& samples() const { return _samples; }

            /// return the current iteration number
            int current_iteration() const { return _current_iteration; }

            int total_iterations() const { return _total_iterations; }

            /// Add a new sample / observation pair
            /// - does not update the model!
            /// - we don't add NaN and inf observations
            void add_new_sample(const Eigen::VectorXd& s, const Eigen::VectorXd& v)
            {
                if (tools::is_nan_or_inf(v))
                    throw EvaluationError();
                _samples.push_back(s);
                _observations.push_back(v);
            }

            /// Evaluate a sample and add the result to the 'database' (sample / observations vectors) -- it does not update the model
            template <concepts::StateFunc StateFunction>
            void eval_and_add(const StateFunction& seval, const Eigen::VectorXd& sample)
            {
                this->add_new_sample(sample, seval(sample));
            }

        private:
            template <concepts::StateFunc StateFunction, concepts::AggregatorFunc AggregatorFunction>
            void _init(const StateFunction& seval, const AggregatorFunction& afun, bool reset = true)
            {
                this->_current_iteration = 0;
                if (reset) {
                    this->_total_iterations = 0;
                    this->_samples.clear();
                    this->_observations.clear();
                }

                if (this->_total_iterations == 0)
                    init_function_t()(seval, afun, *this);
            }

            template <typename BO, typename AggregatorFunction>
            bool _stop(const BO& bo, const AggregatorFunction& afun) const
            {
                stop::ChainCriteria<BO, AggregatorFunction> chain(bo, afun);
                return boost::fusion::accumulate(_stopping_criteria, false, chain);
            }

            template <typename BO, typename AggregatorFunction>
            void _update_stats(BO& bo, const AggregatorFunction& afun)
            { // not const, because some stat class
                // modify the optimizer....
                boost::fusion::for_each(stat_, RefreshStat_f<BO, AggregatorFunction>(bo, afun));
            }

            void _make_res_dir()
            {
                if (!Params::bayes_opt_boptimizer::stats_enabled())
                    return;
                _res_dir = tools::hostname() + "_" + tools::date() + "_" + tools::getpid();
                std::filesystem::path my_path(_res_dir);
                std::filesystem::create_directory(my_path);
            }

            std::string _res_dir;
            int _current_iteration;
            int _total_iterations;
            stopping_criteria_t _stopping_criteria;
            stat_t stat_;
            std::vector<Eigen::VectorXd> _observations;
            std::vector<Eigen::VectorXd> _samples;
            model_t _model;
        };

        namespace _default_hp {
            template <typename Params>
            using model_t = model::GPOpt<Params>;
            template <typename Params>
            using acqui_t = acqui::UCB<typename Params::acqui_ucb, model_t<Params>>;
        }

        /// A shortcut for a BOptimizer with UCB + GPOpt
        /// The acquisition function and the model CANNOT be tuned (use BOptimizer for this)
        template <class Params,
			typename init_t = init::RandomSampling<typename Params::init_randomsampling>,
    		typename StoppingCriteria = boost::fusion::vector<stop::MaxIterations<typename Params::stop_maxiterations>>,
    		typename Stat =  boost::fusion::vector<stat::Samples, stat::AggregatedObservations, stat::ConsoleSummary>,
			typename acqui_opt_t = typename defaults<Params>::acquiopt_t>
        using BOptimizerHPOpt = BOptimizer<Params, _default_hp::model_t<Params>, _default_hp::acqui_t<Params>, init_t, StoppingCriteria, Stat, acqui_opt_t>;
    }
}
#endif
