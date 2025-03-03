#pragma once
#include <limbo/concepts.hpp>
#include <limbo/stat/stat_base.hpp>

namespace limbo::stat
{
	//This class acts as an adapter between limbo's all-static registration of stats functions. Allows registering a dynamic function to be called.
	struct RuntimeStatsFunction : StatBase
	{
		using FuncT = std::function<void(double, Eigen::VectorXd, const std::vector<double>&, const std::vector<Eigen::VectorXd>&)>;

		void addOutputFunction(FuncT outputFunc)
		{
			outFuncs_.emplace_back(std::move(outputFunc));
		}

		template <limbo::concepts::BayesOptimizer BO>
		void operator()(BO const& bo)
		{
			auto [bestObs, bestSample] = bo.model().best_observation();
			for (auto& func : outFuncs_) {
				func(bestObs, bestSample, bo.model().observations(), bo.model().samples());
			}
		}

	private:
		std::vector<FuncT> outFuncs_;
	};

	static_assert(limbo::concepts::StatsFunc<RuntimeStatsFunction>);
}