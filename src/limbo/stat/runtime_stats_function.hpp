#pragma once
#include <limbo/concepts.hpp>
#include <limbo/stat/stat_base.hpp>

namespace limbo::stat
{
	//This class acts as an adapter between limbo's all-static registration of stats functions. Allows registering a dynamic function to be called.
	struct RuntimeStatsFunction : StatBase
	{
		using FuncT = std::function<void(double, Eigen::VectorXd, const std::vector<Eigen::VectorXd>&, const std::vector<Eigen::VectorXd>&)>;

		void addOutputFunction(FuncT outputFunc)
		{
			outFuncs_.emplace_back(std::move(outputFunc));
		}

		template <limbo::concepts::BayesOptimizer BO>
		void operator()(BO const& bo)
		{
			for (auto& func : outFuncs_) {
				func(bo.best_observation(), bo.best_sample(), bo.observations(), bo.samples());
			}
		}

	private:
		std::vector<FuncT> outFuncs_;
	};

	static_assert(limbo::concepts::StatsFunc<RuntimeStatsFunction>);
}