#pragma once
#include <limbo/concepts.hpp>
#include <filesystem>
#include <cnpy.h>
#include "CartesianGenerator.hpp"
#include <tbb/tbb.h>

namespace limbo::serialize
{

    // A function that takes a coordinate vector as input and returns a double.
    template<typename T>
    concept IterFunc = concepts::Callable<T, double, std::vector<unsigned>>;

    enum FunctionFlag
    { // Indicates the type of function to export
        GaussianProcess = 1, // Export mean and sigma of the gaussian process
        LogLikelihood = 2 // export the log likelihood over the hyperparameter space
    };

    inline FunctionFlag operator|(FunctionFlag a, FunctionFlag b)
    {
        return static_cast<FunctionFlag>(static_cast<int>(a) | static_cast<int>(b));
    }


    template<concepts::Model Model>
    void exportFunction(std::filesystem::path const& directory, FunctionFlag flags, Model const& model, size_t samplesPerDim, std::optional<std::function<void(std::string)>> progressCB = std::nullopt)
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
            for (size_t i = 0; i < cartGen.totalIterations(); i++)
            {
                out.push_back(func(cartGen.currentIndex()));
                cartGen.iterate();
            }
            cnpy::npy_save(fname, out.data(), std::vector<size_t>(dimensions, samplesPerDim));
        };

        if (flags & GaussianProcess)
        { // If exporting both then it can be done more efficiently at once

            //This worker runs in sequence generating coordinates to test at.
            class CoordGenerator
            {
            public:
                CoordGenerator(CartesianGenerator* gen) :
                    gen_(gen)
                {}

                std::vector<unsigned> operator()(tbb::flow_control& fc) const
                {
                    if (iter_ > 0)
                    {
                        gen_->iterate();
                        if (gen_->currentIteration() == 0)
                        { // If this happens we've reached the end of the iteration
                            fc.stop();
                            return {};
                        }

                    }
                    ++iter_;
                    return gen_->currentIndex();
                }
            private:
                mutable size_t iter_ = 0;
                CartesianGenerator* gen_;
            };

            //This worker runs in parallel, querying model values at a given coordinate
            class QueryCoord
            {
            public:
                QueryCoord(Model const& model, int samplesPerDim):
					model_(model),
					samplesPerDim_(samplesPerDim)
				{}

                std::tuple<double, double> operator()(std::vector<unsigned> const& coord) const
                {
                    Eigen::VectorXd x(coord.size());
                    for (int i = 0; i < coord.size(); i++)
                    {
                        x(i) = static_cast<double>(coord.at(i)) / (samplesPerDim_ - 1);
                    }
                    auto [mu, sigma_sq] = model_.query(x);
                    return { mu, sigma_sq };
                }

            private:
                Model const& model_;
                int samplesPerDim_;
            };

            // This worker receives values from `QueryCoord` and stores them to be saved later
            class ReceiveGPVals
            {
            public:
                ReceiveGPVals(std::vector<double>* muOut_, std::vector<double>* sigOut_, size_t total_iters, std::optional<std::function<void(std::string)>>& progressCB) :
					muOut(muOut_),
					sigOut(sigOut_),
					prog_(progressCB),
					total_iters_(total_iters)
            	{}

                void operator()(std::tuple<double, double> const& vals) const
                {
                    muOut->push_back(std::get<0>(vals));
                    sigOut->push_back(std::get<1>(vals));
                    if (prog_ && muOut->size())
                    {
                        double percent = static_cast<double>(muOut->size()) / total_iters_ * 100;
                        if (percent >= lastPercent_ + 5)
                        {
                            prog_.value()(std::format("{:d}% Complete", static_cast<int>(percent)));
                            lastPercent_ = percent;
                        }
                    }
                }

            private:
                std::vector<double>* muOut;
                std::vector<double>* sigOut;
                mutable double lastPercent_ = 0;
                size_t total_iters_;
                std::optional<std::function<void(std::string)>>& prog_;
            };

            if (progressCB)
            {
                progressCB.value()("Exporting Gaussian Process");
            }

            CartesianGenerator gen(std::vector<unsigned>(model.dim_in(), samplesPerDim));
            std::vector<double> muOut;
            std::vector<double> sigOut;
            muOut.reserve(gen.totalIterations());
            sigOut.reserve(gen.totalIterations());

            // Use TBB pipeline to evaluate GP values in parallel with order maintained
            tbb::filter<void, std::vector<unsigned>> f1(tbb::filter_mode::serial_in_order, CoordGenerator(&gen));
            tbb::filter<std::vector<unsigned>, std::tuple<double, double>> f2(tbb::filter_mode::parallel, QueryCoord(model, samplesPerDim));
            tbb::filter<std::tuple<double, double>, void> f3(tbb::filter_mode::serial_in_order, ReceiveGPVals(&muOut, &sigOut, gen.totalIterations(), progressCB));
            tbb::filter<void, void> allFilters = f1 & f2 & f3;
            tbb::parallel_pipeline(std::thread::hardware_concurrency(),
               allFilters);

            // Save the files
            cnpy::npy_save((directory / "mean.npy").string(), muOut.data(), std::vector<size_t>(model.dim_in(), samplesPerDim));
            cnpy::npy_save((directory / "kernel.npy").string(), sigOut.data(), std::vector<size_t>(model.dim_in(), samplesPerDim));
            std::ofstream(directory / "mean.txt") << 0 << "," << 1 << "\n";
            std::ofstream(directory / "kernel.txt") << 0 << "," << 1 << "\n";
        }
		
		if (flags & LogLikelihood)
		{
            if (progressCB)
            {
                progressCB.value()("Exporting Log Likelihood function");
            }
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
