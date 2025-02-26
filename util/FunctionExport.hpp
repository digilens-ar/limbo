#pragma once
#include <limbo/concepts.hpp>
#include <filesystem>
#include <cnpy.h>
#include "CartesianGenerator.hpp"

namespace limbo::serialize
{

    // A function that takes a coordinate vector as input and returns a double.
    template<typename T>
    concept IterFunc = concepts::Callable<T, double, std::vector<unsigned>>;

    enum FunctionFlag
    { // Indicates the type of function to export
        MeanFunction = 1,
        KernelFunction = 2,
        LogLikelihood = 4
    };

    FunctionFlag operator|(FunctionFlag a, FunctionFlag b)
    {
        return static_cast<FunctionFlag>(static_cast<int>(a) | static_cast<int>(b));
    }

    template<concepts::Model Model>
    void exportFunction(std::filesystem::path const& directory, FunctionFlag flags, Model const& model, size_t samplesPerDim)
	{
		if (!exists(directory))
		{
			create_directories(directory);
		}

        auto saveBinary = []<typename IterFunc>(std::string const& fname, int dimensions, int samplesPerDim, IterFunc func)
        {
            std::vector<double> out;
            out.reserve(dimensions * samplesPerDim);
            CartesianGenerator cartGen(std::vector<unsigned>(dimensions, samplesPerDim));
            for (unsigned i = 0; i < cartGen.totalIterations(); i++)
            {
                out.push_back(func(cartGen.currentIndex()));
                cartGen.iterate();
            }
            cnpy::npy_save(fname, out.data(), std::vector<size_t>(dimensions, samplesPerDim));
        };

		if (flags & MeanFunction)
		{
            saveBinary((directory / "mean.npy").string(), model.dim_in(), samplesPerDim, [samplesPerDim, &model](std::vector<unsigned> const& coord) -> double
            {
                Eigen::VectorXd x(coord.size());
                for (int i=0; i<coord.size(); i++)
                {
                    x(i) = static_cast<double>(coord.at(i)) / (samplesPerDim - 1);
                }
                return model.mu(x);
            });
            std::ofstream(directory / "mean.txt") << 0 << "," << 1 << "\n";
		}
		if (flags & KernelFunction)
		{
            saveBinary((directory / "kernel.npy").string(), model.dim_in(), samplesPerDim, [samplesPerDim, &model](std::vector<unsigned> const& coord) -> double
                {
                    Eigen::VectorXd x(coord.size());
                    for (int i = 0; i < coord.size(); i++)
                    {
                        x(i) = static_cast<double>(coord.at(i)) / (samplesPerDim - 1);
                    }
                    return model.sigma_sq(x);
                });
            std::ofstream(directory / "kernel.txt") << 0 << "," << 1 << "\n";
		}
		if (flags & LogLikelihood)
		{
            constexpr double LIMIT = 8;
            Model copy(model); // Create a copy so that we don't have to modify the original with hyperparameter adjustments.
            saveBinary((directory / "loglik.npy").string(), copy.kernel_function().h_params_size(), samplesPerDim, [samplesPerDim, &copy](std::vector<unsigned> const& coord) -> double
                {
                    size_t dims = copy.kernel_function().h_params_size();
                    Eigen::VectorXd x(dims);
                    for (int i = 0; i < dims; i++)
                    {
                        x(i) = 2 * LIMIT * (static_cast<double>(coord.at(i)) / (samplesPerDim - 1) - 0.5);
                    }
                    copy.set_kernel_hyperparams(x);
                    return copy.compute_log_lik();
                });
            std::ofstream(directory / "loglik.txt") << -LIMIT << "," <<  LIMIT << "\n";
		}
	}
}
