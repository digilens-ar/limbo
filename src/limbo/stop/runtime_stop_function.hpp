#pragma once
#include <limbo/concepts.hpp>

namespace limbo::stop
{
	//This class acts as an adapter between limbo's all-static registration of stop conditions. Allows registering a dynamic function to be called.
	struct RuntimeStopFunction
	{
		using StopFuncT = std::function<bool(double, Eigen::VectorXd, std::vector<double>, std::vector<Eigen::VectorXd>, std::string&)>;

		template<concepts::BayesOptimizer BO>
		bool operator()(BO const& bo, std::string& stopMessage) const
		{
			auto const& gaussianProcess = bo.model();
			auto [bestObs, bestSample] = gaussianProcess.best_observation();
			for (auto const& func : stopFuncs_)
			{
				if (func(bestObs, bestSample, gaussianProcess.observations(), gaussianProcess.samples(), stopMessage))
				{
					return true;
				}
			}
			return false;
		}

		void addStopFunction(StopFuncT stopFunc)
		{
			stopFuncs_.emplace_back(std::move(stopFunc));
		}

	private:
		std::vector<StopFuncT> stopFuncs_;
	};

	static_assert(concepts::StoppingCriteria<RuntimeStopFunction>);
}