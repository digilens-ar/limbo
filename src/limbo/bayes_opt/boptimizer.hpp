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
#include <iterator>
#include <filesystem>
#include <Eigen/Core>
#include <boost/fusion/container.hpp>
#include <boost/fusion/algorithm.hpp>
#include <limbo/public.hpp>
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

namespace limbo {
    namespace defaults {
        struct bayes_opt_boptimizer {
            BO_PARAM(int, hp_period, -1); // If this is a positive number the model will `optimize_hyperparameters` every `hp_period + i * hp_period_scaler` iterations
            BO_PARAM(double, hp_period_scaler, 0.0); // If this is 0 the hyperparameter optimization will occur on a regular schedule. If it is positive then the frequency of optimization will decrease over time. This can be useful since optimization becomes time consuming when more samples are present and usually doesn't result in major changes of the hyperparameters if the optimization has already been run a few times earlier. Values must be less than 1. Reccommended values are ~0.0-0.2 
        	BO_PARAM(bool, stats_enabled, true);
            BO_PARAM(bool, bounded, true);
        };
    }

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

		+----------------+-------------+---------+---------------+
		|type            |typedef      | argument| default       |
		+================+=============+=========+===============+
		|init. func.     |InitializerT | initfun | RandomSampling|
		+----------------+-------------+---------+---------------+
		|model           |model_t      | modelfun| GP<...>       |
		+----------------+-------------+---------+---------------+
		|acquisition fun.|aqui_t       | acquifun| GP_UCB        |
		+----------------+-------------+---------+---------------+
		|statistics      | stat_t      | statfun | see below     |
		+----------------+-------------+---------+---------------+
		|stopping crit.  | stop_t      | stopcrit| MaxIterations |
		+----------------+-------------+---------+---------------+
		|acqui. optimizer|acquiopt_t   | acquiopt | see below |
        +----------------+-------------+----------+---------------+
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
            BOptimizer(int dimIn);

            BOptimizer(const BOptimizer& other) = delete; // copy is disabled (dangerous and useless)
            BOptimizer& operator=(const BOptimizer& other) = delete; // copy is disabled (dangerous and useless)
            BOptimizer(BOptimizer&& other) = default;
            BOptimizer& operator=(BOptimizer&& other) = default;

            template <typename Archive>
            void loadFromArchive(Archive const& archive );

            /// The main function (run the Bayesian optimization algorithm)
            template <concepts::StateFunc StateFunction>
            std::string optimize(const StateFunction& sfun, bool reset = true, std::optional<Eigen::VectorXd> const& initialPoint = std::nullopt);

            /// return the best observation so far (i.e. max(f(x)))
            double best_observation() const;

            /// return the best sample so far (i.e. the argmax(f(x)))
            const Eigen::VectorXd& best_sample() const;

            model_type const& model() const;

            /// return the vector of points of observations (observations can be multi-dimensional, hence the VectorXd) -- f(x)
            std::vector<double> const& observations() const;

            /// return the list of the points that have been evaluated so far (x)
            std::vector<Eigen::VectorXd> const& samples() const;

            /// return the current iteration number
            int total_iterations() const;

            template<concepts::EvalFunc EvalFunc>
            Eigen::VectorXd optimizeFunction(const EvalFunc& evalFunc, Eigen::VectorXd const& initPoint, std::optional<std::vector<std::pair<double, double>>> const& bounds) const;

            /// Evaluate a sample and add the result to the 'database' (sample / observations vectors)
            template <concepts::StateFunc StateFunction>
            EvaluationStatus eval_and_add(const StateFunction& seval, const Eigen::VectorXd& sample);

            bool isBounded() const;

            void addInequalityConstraint(constraint_func_t func);

            void addEqualityConstraint(constraint_func_t func);

            bool hasConstraints() const;

            void setStatsOutputDirectory(std::filesystem::path const& dir);

        protected:
            typename boost::mpl::if_<boost::fusion::traits::is_sequence<StoppingCriteria>, StoppingCriteria, boost::fusion::vector<StoppingCriteria>>::type stopping_criteria_;
            stats_t  stat_;
            std::filesystem::path outputDir_;
        private:
            size_t _total_iterations = 0;
            size_t iterations_since_hp_optimize_ = 0;
            acqui_opt_t acqui_optimizer;
            std::vector<constraint_func_t> equalityConstraints_;
            std::vector<constraint_func_t> inequalityConstraints_;
            model_type _model;
        };


        /// A shortcut for a BOptimizer with UCB + GPOpt
        /// The acquisition function and the model CANNOT be tuned (use BOptimizer for this)
        template <class Params,
			typename InitializerT = init::RandomSampling<typename Params::init_randomsampling>,
    		typename StoppingCriteriaT = boost::fusion::vector<stop::MaxIterations<typename Params::stop_maxiterations>>,
    		typename StatT =  boost::fusion::vector<stat::Samples, stat::AggregatedObservations, stat::ConsoleSummary>,
			typename acqui_opt_t = typename defaults<Params>::acquiopt_t>
        using BOptimizerHPOpt = BOptimizer<
            Params,
            model::GPOpt<Params>,
            acqui::UCB<typename Params::acqui_ucb, model::GPOpt<Params>>,
    		InitializerT,
    		StoppingCriteriaT,
    		StatT,
    		acqui_opt_t>;

    }
}

// Include the implementation of boptimizers methods
#include "boptimizer_impl.hpp"

#endif
