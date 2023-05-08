#pragma once
#include <limbo/concepts.hpp>

namespace limbo::stat
{
	//This class acts as an adapter between limbo's all-static registration of stats functions. Allows registering a dynamic function to be called.
	struct RuntimeStatsFunction
	{
		using FuncT = std::function<void(double, Eigen::VectorXd, const std::vector<Eigen::VectorXd>&, const std::vector<Eigen::VectorXd>&)>;

		void addOutputFunction(FuncT outputFunc)
		{
			outFuncs_.emplace_back(std::move(outputFunc));
		}

		template <limbo::concepts::BayesOptimizer BO, limbo::concepts::AggregatorFunc AggFunc>
		void operator()(BO const& bo, AggFunc const& aggFunc)
		{
			for (auto& func : outFuncs_) {
				func(aggFunc(bo.best_observation(aggFunc)), bo.best_sample(), bo.observations(), bo.samples());
			}
		}

	private:
		std::vector<FuncT> outFuncs_;
	};

	static_assert(limbo::concepts::StatsFunc<RuntimeStatsFunction>);
}