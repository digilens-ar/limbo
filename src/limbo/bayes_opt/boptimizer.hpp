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

#define SAVE_HP_MODELS

#include <algorithm>
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
#include "limbo/stop/max_iterations.hpp"
#ifdef USE_NLOPT
#include <limbo/opt/nlopt_no_grad.hpp>
#elif defined USE_LIBCMAES
#include <limbo/opt/cmaes.hpp>
#else
#include <limbo/opt/grid_search.hpp>
#endif
#include <limbo/serialize/text_archive.hpp>
#include <filesystem>
#include <fstream>


namespace limbo {
    namespace defaults {
        struct bayes_opt_boptimizer {
            BO_PARAM(int, hp_period, -1); // If this is a positive number the model will `optimize_hyperparameters` every `hp_period` iterations
            BO_PARAM(bool, stats_enabled, true);
            BO_PARAM(bool, bounded, true);
        };
    }

    class EvaluationError : public std::exception
    {
    public:
        using std::exception::exception;
    };

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
			concepts::AcquisitionFunc acqui_t = acqui::UCB<typename Params::acqui_ucb, model_type>,
			typename init_t = init::RandomSampling<typename Params::init_randomsampling>,
    		typename StoppingCriteria = boost::fusion::vector<stop::MaxIterations<typename Params::stop_maxiterations>>,
    		typename Stat =  boost::fusion::vector<stat::Samples, stat::AggregatedObservations, stat::ConsoleSummary>,
			concepts::Optimizer acqui_opt_t = typename defaults<Params>::acquiopt_t
    	>
        class BOptimizer {
        public:
            // Public types
            using acquisition_function_t = acqui_t;
            using init_function_t = init_t;
            using stopping_criteria_t = StoppingCriteria;
            using model_t = model_type;
            using stats_t = typename boost::mpl::if_<boost::fusion::traits::is_sequence<Stat>, Stat, boost::fusion::vector<Stat>>::type;
            using constraint_func_t = std::function<std::pair<double, std::optional<Eigen::VectorXd>>(Eigen::VectorXd, bool)>;

            /// default constructor
            BOptimizer(int dimIn):
				acqui_optimizer(acqui_opt_t::create(dimIn)),
				_model(dimIn)
            {}

            BOptimizer(const BOptimizer& other) = delete; // copy is disabled (dangerous and useless)
            BOptimizer& operator=(const BOptimizer& other) = delete; // copy is disabled (dangerous and useless)

            /// The main function (run the Bayesian optimization algorithm)
            template <concepts::StateFunc StateFunction>
            std::string optimize(const StateFunction& sfun, bool reset = true)
            {
                this->_current_iteration = 0;
                if (reset) {
                    this->_total_iterations = 0;
                    this->_samples.clear();
                    this->_observations.clear();
                }

                if (this->_total_iterations == 0) {
                    EvaluationStatus initStatus = init_t()(sfun, *this);
                    if (initStatus == TERMINATE)
                    {
                        return "Initialization requested that optimization be terminated";
                    }
                }

                if (!this->_observations.empty())
                    _model.initialize(this->_samples, this->_observations);
                else
                    _model = model_type(sfun.dim_in());

                if (Params::bayes_opt_boptimizer::hp_period() > 0 && _observations.size() >= Params::bayes_opt_boptimizer::hp_period())
                { // If the initialization includes enough samples for hyper parameter optimization then run it. TODO untested change
					#ifdef SAVE_HP_MODELS
                    _model.save(serialize::TextArchive((outputDir_ / "modelArchive_init").string()));
					#endif
                	_model.optimize_hyperparams();
					#ifdef SAVE_HP_MODELS
					_model.save(serialize::TextArchive((outputDir_ / "modelArchive_post_init").string()));
					#endif
                }

                std::string stopMessage = "";
                // While no stopping criteria return `true`
                while (!boost::fusion::accumulate(_stopping_criteria, false, [this, &stopMessage](bool state, concepts::StoppingCriteria auto const& stop_criteria) { return state || stop_criteria(*this, stopMessage); }))
                {
                    acquisition_function_t acqui(_model, this->_current_iteration);

                    Eigen::VectorXd starting_point = tools::random_vector(sfun.dim_in(), Params::bayes_opt_boptimizer::bounded());
                    Eigen::VectorXd new_sample = acqui_optimizer.optimize(
                        [&](const Eigen::VectorXd& x, bool g) -> opt::eval_t { return acqui(x, g); },
                        starting_point, 
                        Params::bayes_opt_boptimizer::bounded());

                	auto status = this->eval_and_add(sfun, new_sample);
                    if (status == TERMINATE)
                    {
                        stopMessage = "Objective function requested that optimization be terminated";
                        break;
                    }

                    if (Params::bayes_opt_boptimizer::stats_enabled()) {
                        //update stats
                        boost::fusion::for_each(
                            stat_, 
                            [this](concepts::StatsFunc auto& func)
                            {
	                            func.template operator()<decltype(*this)>(*this);
                            });
                    }

                    _model.add_sample(this->_samples.back(), this->_observations.back()); // update the model

                    if (Params::bayes_opt_boptimizer::hp_period() > 0
                        && (this->_current_iteration + 1) % Params::bayes_opt_boptimizer::hp_period() == 0) {
                        _model.optimize_hyperparams();
#ifdef SAVE_HP_MODELS
                        _model.save(serialize::TextArchive((outputDir_ / ("modelArchive_" + std::to_string(_current_iteration))).string()));
#endif
                    }


                	++this->_current_iteration;
                    ++this->_total_iterations;
                }
                return stopMessage;
            }

            /// return the best observation so far (i.e. max(f(x)))
            double best_observation() const
            {
                auto max_e = std::max_element(_observations.begin(), _observations.end());
                return this->_observations[std::distance(_observations.begin(), max_e)];
            }

            /// return the best sample so far (i.e. the argmax(f(x)))
            const Eigen::VectorXd& best_sample() const
            {
                auto max_e = std::max_element(_observations.begin(), _observations.end());
                return this->_samples[std::distance(_observations.begin(), max_e)];
            }

            model_type const& model() const { return _model; }
            stats_t& statsFunctors() { return stat_; }

            /// return the vector of points of observations (observations can be multi-dimensional, hence the VectorXd) -- f(x)
            std::vector<double> const& observations() const { return _observations; }

            /// return the list of the points that have been evaluated so far (x)
            std::vector<Eigen::VectorXd> const& samples() const { return _samples; }

            /// return the current iteration number
            int current_iteration() const { return _current_iteration; }

            int total_iterations() const { return _total_iterations; }

            acqui_opt_t const& acquisitionOptimizer() const { return acqui_optimizer; }

            /// Evaluate a sample and add the result to the 'database' (sample / observations vectors) -- it does not update the model
            template <concepts::StateFunc StateFunction>
            EvaluationStatus eval_and_add(const StateFunction& seval, const Eigen::VectorXd& sample)
            {
                auto [status, observation] = seval(sample);
                if (status == OK) // TODO if `seval` returns `SKIP` we need to do something to avoid that sample being tested again. I.E addd a very negative observation
                {
                    /// Add a new sample / observation pair
					/// - does not update the model!
					/// - we don't add NaN and inf observations
					if (std::isnan(observation))
					{
                        throw EvaluationError("Merit function returned a NaN value");
					}
                    if (std::isinf(observation)) 
                    {
                        throw EvaluationError("Merit function returned an infinite value");
                    }
                    _samples.push_back(sample);
                    _observations.push_back(observation);
                }
                return status;
            }

            bool isBounded() const { return Params::bayes_opt_boptimizer::bounded(); }

            template<concepts::EvalFunc Func>
            void addInequalityConstraint(Func func)
            {
                auto& it = inequalityConstraints_.emplace_back(std::make_unique<constraint_func_t>(std::move(func)));
                acqui_optimizer.add_inequality_constraint(it.get());
            }

            template<concepts::EvalFunc Func>
            void addEqualityConstraint(Func func)
            {
                auto& it = equalityConstraints_.emplace_back(std::make_unique<constraint_func_t>(std::move(func)));
                acqui_optimizer.add_equality_constraint(it.get());
            }

            bool hasConstraints() const
            {
                return !equalityConstraints_.empty() || !inequalityConstraints_.empty();
            }

            bool constraintsAreSatisfied(Eigen::VectorXd const& sampleLocation) const
            {
	            for (auto const& ineq : inequalityConstraints_)
	            {
                    auto [val, gradient] = ineq->operator()(sampleLocation, false);
                    if (val >= 0)
                    {
                        return false;
                    }
	            }
                for (auto const& eq : equalityConstraints_)
                {
                    auto [val, gradient] = eq->operator()(sampleLocation, false);
                    if (std::abs(val) > 1e-8) // Ideally the value should be 0 but we use 1e-8 to give a little wiggle room.
                    {
                        return false;
                    }
                }
                return true;
            }

            void setStatsOutputDirectory(std::filesystem::path const& dir)
            {
                assert(exists(dir));
                assert(std::filesystem::is_directory(dir));
                outputDir_ = dir;
                boost::fusion::for_each(stat_, [&dir](auto& stat) {stat.setOutputDirectory(dir); });
            }

        private:

            int _current_iteration = 0;
            int _total_iterations = 0;
            typename boost::mpl::if_<boost::fusion::traits::is_sequence<StoppingCriteria>, StoppingCriteria, boost::fusion::vector<StoppingCriteria>>::type _stopping_criteria;
            stats_t  stat_;
            acqui_opt_t acqui_optimizer;
			std::vector<double> _observations;
            std::vector<Eigen::VectorXd> _samples;
            std::vector<std::unique_ptr<constraint_func_t>> equalityConstraints_;
            std::vector<std::unique_ptr<constraint_func_t>> inequalityConstraints_;
            model_type _model;
            std::filesystem::path outputDir_;
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
