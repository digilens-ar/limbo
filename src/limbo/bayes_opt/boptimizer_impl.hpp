#pragma once

#define SAVE_HP_MODELS

#ifdef SAVE_HP_MODELS
#include <limbo/serialize/text_archive.hpp>
#include <filesystem>
#include <fstream>
#include <chrono>
#endif

namespace limbo::bayes_opt
{

//Shorten the definition of the template class
#define OptClass(returnType)  template <class Params, concepts::Model model_type, concepts::AcquisitionFunc acqui_t, typename init_t, typename \
			StoppingCriteria, typename Stat, concepts::Optimizer acqui_opt_t> \
	returnType BOptimizer<Params, model_type, acqui_t, init_t, StoppingCriteria, Stat, acqui_opt_t>

//Shorten the definition of a template function of the template class
#define OptClassTemplateFunction(templateStatement, returnType)  template <class Params, concepts::Model model_type, concepts::AcquisitionFunc acqui_t, typename init_t, typename \
			StoppingCriteria, typename Stat, concepts::Optimizer acqui_opt_t> \
			templateStatement \
			returnType BOptimizer<Params, model_type, acqui_t, init_t, StoppingCriteria, Stat, acqui_opt_t>


	template <class Params, concepts::Model model_type, concepts::AcquisitionFunc acqui_t, typename init_t,
		typename StoppingCriteria, typename Stat, concepts::Optimizer acqui_opt_t> 
	BOptimizer<Params, model_type, acqui_t, init_t, StoppingCriteria, Stat, acqui_opt_t>::BOptimizer(int dimIn) :
		acqui_optimizer(acqui_opt_t::create(dimIn)),
		_model(dimIn)
	{}

	OptClassTemplateFunction(template <typename Archive>, void)::loadFromArchive(
		Archive const& archive)
	{
		_model = model_type::load(archive);
		_total_iterations = _model.observations().size();
	}

	OptClassTemplateFunction(template <concepts::StateFunc StateFunction>, std::string)::optimize(
		StateFunction const& sfun, bool reset, std::optional<Eigen::VectorXd> const& initialPoint)
	{
		if (reset) {
			_total_iterations = 0;
			_model = model_type(sfun.dim_in());
		}

		if (_total_iterations == 0) {
			EvaluationStatus initStatus;
			if (initialPoint.has_value())
			{
				initStatus = eval_and_add(sfun, initialPoint.value());
				if (initStatus == TERMINATE)
				{
					return "Initialization requested that optimization be terminated";
				}
			}
			initStatus = init_t()(sfun, *this);
			if (initStatus == TERMINATE)
			{
				return "Initialization requested that optimization be terminated";
			}
		}

		if (Params::bayes_opt_boptimizer::hp_period() > 0)
		{ // If hyperparameter tuning is enabled then run it after initialization
#ifdef SAVE_HP_MODELS
                _model.save(serialize::TextArchive((outputDir_ / "modelArchive_init").string()));
#endif
			_model.optimize_hyperparams();
#ifdef SAVE_HP_MODELS
				_model.save(serialize::TextArchive((outputDir_ / "modelArchive_post_init").string()));
#endif
		}

		std::optional<std::vector<std::pair<double, double>>> parameterBounds = std::nullopt;
		if (Params::bayes_opt_boptimizer::bounded())
		{
			parameterBounds = std::vector<std::pair<double, double>>(_model.dim_in(), std::make_pair( 0.0, 1.0 ));
		}

		std::string stopMessage = "";
		// While no stopping criteria return `true`
		while (!boost::fusion::accumulate(stopping_criteria_, false, [this, msgPtr=&stopMessage](bool state, concepts::StoppingCriteria auto const& stop_criteria) { return state || stop_criteria(*this, *msgPtr); }))
		{
			acquisition_function_t acqui(_model, this->_total_iterations);

			Eigen::VectorXd starting_point = tools::random_vector(sfun.dim_in(), Params::bayes_opt_boptimizer::bounded());
			Eigen::VectorXd new_sample = acqui_optimizer.optimize(
				[&](const Eigen::VectorXd& x, bool g) -> opt::eval_t { return acqui(x, g); },
				starting_point, 
				parameterBounds);

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

			if (Params::bayes_opt_boptimizer::hp_period() > 0)
			{
				if ((iterations_since_hp_optimize_ + 1) % static_cast<int>(Params::bayes_opt_boptimizer::hp_period() + _total_iterations * Params::bayes_opt_boptimizer::hp_period_scaler()) == 0)
				{
					iterations_since_hp_optimize_ = 0;
					const auto start = std::chrono::high_resolution_clock::now();
					_model.optimize_hyperparams();
					spdlog::info("Hyperparameter optimization for iteration {} took {} ms", _total_iterations, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count());
#ifdef SAVE_HP_MODELS
                        _model.save(serialize::TextArchive((outputDir_ / ("modelArchive_" + std::to_string(_total_iterations))).string()));
#endif
				}
				else
				{
					++iterations_since_hp_optimize_;
				}
			}

			++_total_iterations;
		}
		return stopMessage;
	}

	OptClass(double)::best_observation() const
	{
		auto max_e = std::max_element(observations().begin(), observations().end());
		return observations()[std::distance(observations().begin(), max_e)];
	}

	OptClass(Eigen::VectorXd const&)::best_sample() const
	{
		auto max_e = std::max_element(observations().begin(), observations().end());
		return samples()[std::distance(observations().begin(), max_e)];
	}

	OptClass(model_type const&)::model() const
	{ return _model; }

	OptClass(std::vector<double> const&)::observations() const
	{ return _model.observations(); }

	OptClass(std::vector<Eigen::VectorXd> const&)::samples() const
	{ return _model.samples(); }

	OptClass(int)::total_iterations() const
	{ return _total_iterations; }

	OptClassTemplateFunction(template <concepts::EvalFunc EvalFunc>, Eigen::VectorXd)::optimizeFunction(
		const EvalFunc& evalFunc, const Eigen::VectorXd& initPoint,
		std::optional<std::vector<std::pair<double, double>>> const& bounds) const
	{
		return acqui_optimizer.optimize(evalFunc, initPoint, bounds);
	}

	OptClassTemplateFunction(template <concepts::StateFunc StateFunction>, EvaluationStatus)::eval_and_add(const StateFunction& seval, const Eigen::VectorXd& sample)
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
			_model.add_sample(sample, observation);
		}
		return status;
	}

	OptClass(bool)::isBounded() const
	{ return Params::bayes_opt_boptimizer::bounded(); }

	OptClass(void)::addInequalityConstraint(constraint_func_t func)
	{
		auto& it = inequalityConstraints_.emplace_back(std::move(func));
		acqui_optimizer.add_inequality_constraint(&it);
	}

	OptClass(void)::addEqualityConstraint(constraint_func_t func)
	{
		auto& it = equalityConstraints_.emplace_back(std::move(func));
		acqui_optimizer.add_equality_constraint(&it);
	}

	OptClass(bool)::hasConstraints() const
	{
		return !equalityConstraints_.empty() || !inequalityConstraints_.empty();
	}

	OptClass(void)::setStatsOutputDirectory(std::filesystem::path const& dir)
	{
		assert(exists(dir));
		assert(std::filesystem::is_directory(dir));
		outputDir_ = dir;
		boost::fusion::for_each(stat_, [&dir](auto& stat) {stat.setOutputDirectory(dir); });
	}
}