#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <random>
#include <limbo/limbo.hpp>

static std::default_random_engine eng;

static std::tuple<Eigen::VectorXd, double> generateObservation(size_t n)
{
	return { Eigen::VectorXd::Random(n, 1), std::uniform_real_distribution<double>(0, 1)(eng)};
}

static void kernelLFOpt(benchmark::State& state)
{
	int numSamples = state.range(0);
	int dim = state.range(1);
	std::vector<Eigen::VectorXd> samples;
	std::vector<double> observations;
	for (int i=0; i<numSamples; i++)
	{
		auto [s, o] = generateObservation(dim);
		samples.push_back(std::move(s));
		observations.push_back(o);
	}

	struct Params
	{
		struct kernel_opt : limbo::defaults::kernel {};
		struct kernel_squared_exp_ard : limbo::defaults::kernel_squared_exp_ard {};
		struct opt_rprop : limbo::defaults::opt_rprop {};
	};

	limbo::model::GP<
		limbo::kernel::SquaredExpARD<Params::kernel_opt, Params::kernel_squared_exp_ard>,
		limbo::mean::Data,
		limbo::model::gp::KernelLFOpt<Params::opt_rprop>
	> gp(dim);

	gp.initialize(samples, observations);
	for (auto _ : state)
	{
		gp.optimize_hyperparams();
	}
}

BENCHMARK(kernelLFOpt)->ArgsProduct({ {50, 100, 200}, {2, 4, 8} })->Unit(benchmark::kMillisecond)->MinTime(3);