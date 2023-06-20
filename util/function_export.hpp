#pragma once
#include <limbo/concepts.hpp>
#include <filesystem>
#include <cnpy.h>

namespace limbo::serialize
{

    class CartesianGenerator
    {
    public:
        CartesianGenerator(std::vector<unsigned> numPerDimension) :
            numPerDimension_(std::move(numPerDimension)),
            currentIdx_(numPerDimension_.size(), 0),
            currentDim_(numPerDimension_.size() - 1),
            currentIteration_(0),
            totalIterations_(1)
        {
            for (auto numThisDim : numPerDimension_)
            {
                totalIterations_ *= numThisDim;
            }
        }

        std::vector<unsigned> const& currentIndex() const
        {
            return currentIdx_;
        }

        void iterate()
        {
            if (currentIdx_.at(currentDim_) == numPerDimension_.at(currentDim_) - 1)
            { // we are at the maximum value for the current axis. Need to set value back to 0 and switch to a different axis.
                currentIdx_.at(currentDim_) = 0;
                if (currentDim_ == 0)
                { // this should only happen once we've iterated through the whole thing and need to roll back to 00000
                    currentDim_ = numPerDimension_.size() - 1; // reset back to first iteration
                    currentIteration_ = 0;
                }
                else {
                    --currentDim_;
                    iterate(); // This will then check the next slowest dimesion
                }
            }
            else
            { // there are more iterations left at this index.
                currentIdx_.at(currentDim_)++;
                currentIteration_++;
                if (currentDim_ < numPerDimension_.size() - 1)
                { // Now that we've iterated the value for our current axis set the axis back to the fastest axis.
                    currentDim_ = numPerDimension_.size() - 1;
                }
            }
        }

        unsigned totalIterations() const
        {
            return totalIterations_;
        }

        unsigned currentIteration() const
        {
            return currentIteration_;
        }
    private:
        std::vector<unsigned> numPerDimension_;
        std::vector<unsigned> currentIdx_;
        unsigned currentDim_;
        unsigned totalIterations_;
        unsigned currentIteration_;
    };

    template<typename T>
    concept IterFunc = concepts::Callable<T, double, std::vector<int>>;

	class FunctionExport
	{
	public:
		enum FunctionFlag
		{
			MeanFunction = 1,
			KernelFunction = 2,
			LogLikelihood = 4

		};

		friend FunctionFlag operator|(FunctionFlag a, FunctionFlag b)
		{
			return static_cast<FunctionFlag>(static_cast<int>(a) | static_cast<int>(b));
		}


        template<concepts::Model Model>
		FunctionExport(std::filesystem::path const& directory, FunctionFlag flags, Model const& model, size_t samplesPerDim)
		{
			if (!exists(directory))
			{
				create_directories(directory);
			}

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
                        copy.kernel_function().set_h_params(x);
                        copy.recompute(false);
                        return copy.compute_log_lik();
                    });
                std::ofstream(directory / "loglik.txt") << -LIMIT << "," <<  LIMIT << "\n";
			}
		}

	private:
        template<typename IterFunc>
		static void saveBinary(std::string const& fname, int dimensions, int samplesPerDim, IterFunc func)
        {
            // std::vector<int> oneDCoords;
            // for (int j = 0; j < samplesPerDim; j++)
            // {
            //     oneDCoords.push_back(j);
            // }
            // std::vector<std::vector<int>> coords(dimensions, oneDCoords);
            std::vector<double> out;
            CartesianGenerator cartGen(std::vector<unsigned>(dimensions, samplesPerDim));
            for (int i=0; i<cartGen.totalIterations(); i++)
            {
                out.push_back(func(cartGen.currentIndex()));
                cartGen.iterate();
            }
            cnpy::npy_save(fname, out.data(), std::vector<size_t>(dimensions, samplesPerDim));
        }
	};
}
