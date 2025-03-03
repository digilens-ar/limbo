#pragma once
#include <limbo/stat/stat_base.hpp>
#include <limbo/serialize/text_archive.hpp>

namespace limbo::stat
{
	/**
	 * \brief Save the model data to a text file and a text archive named by the current iteration. The text file has 4 columns: parameterValue, mu, sigma, acquisitionFunction. Only supports 1d problems.
	 */
	struct ModelExport : StatBase
	{
		template <typename BO>
		void operator()(BO const& bo)
		{
			assert(bo.model().dim_in() == 1);
			std::ofstream f(get_log_directory() / (std::to_string(bo.total_iterations()) + ".dat"));
			using acqFuncT = typename std::decay_t<decltype(bo)>::acquisition_function_t;
			acqFuncT acquisitionFunction(bo.model(), bo.total_iterations());
			for (int i = 0; i < 100; ++i) {
				Eigen::VectorXd v{ { i / 99.0 } };
				auto [mu, sigma_sq] = bo.model().query(v);
				auto [acqVal, grad] = acquisitionFunction(v, false);
				f << v.transpose() << " " << mu[0] << " " << std::sqrt(sigma_sq) << " " << acqVal << std::endl;
			}
			bo.model().save(serialize::TextArchive((get_log_directory() / (std::to_string(bo.total_iterations()) + ".model")).string()));
		}
	};
}