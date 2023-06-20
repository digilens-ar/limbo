#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <random>
#include <limbo/limbo.hpp>

static std::default_random_engine eng;

// see : http://www.sfu.ca/~ssurjano/goldpr.html
// (with ln, as suggested in Jones et al.)
struct GoldenPrice {
	BO_PARAM(size_t, dim_in, 2);

	double operator()(const Eigen::VectorXd& xx) const
	{
		Eigen::VectorXd x = (4.0 * xx).array() - 2.0;
		double r = (1 + (x(0) + x(1) + 1) * (x(0) + x(1) + 1) * (19 - 14 * x(0) + 3 * x(0) * x(0) - 14 * x(1) + 6 * x(0) * x(1) + 3 * x(1) * x(1))) * (30 + (2 * x(0) - 3 * x(1)) * (2 * x(0) - 3 * x(1)) * (18 - 32 * x(0) + 12 * x(0) * x(0) + 48 * x(1) - 36 * x(0) * x(1) + 27 * x(1) * x(1)));

		return -log(r) + 5 ;
	}
};

static std::tuple<Eigen::VectorXd, double> generateObservation(size_t n)
{
	auto sample = Eigen::VectorXd::Random(2, 1);
	return { sample, GoldenPrice()(sample)};
}

namespace
{
	struct Params
	{
		struct kernel_opt : limbo::defaults::kernel {};
		struct kernel_squared_exp_ard : limbo::defaults::kernel_squared_exp_ard {};
		struct opt_rprop : limbo::defaults::opt_rprop
		{
			BO_DYN_PARAM(double, eps_stop);
		};
		struct opt_irpropplus : limbo::defaults::opt_irpropplus
		{
			BO_DYN_PARAM(double, min_gradient);
		};
		struct opt_parallel : limbo::defaults::opt_parallelrepeater
		{
			BO_DYN_PARAM(int, repeats);

			/// epsilon of deviation: init + [-epsilon,epsilon]
			BO_PARAM(double, epsilon, 1);
		};
	};

	BO_DECLARE_DYN_PARAM_INIT(double, Params::opt_rprop, eps_stop, 0.0);
	BO_DECLARE_DYN_PARAM_INIT(double, Params::opt_irpropplus, min_gradient, 0.0);
	BO_DECLARE_DYN_PARAM_INIT(int, Params::opt_parallel, repeats, 10);
}

static std::vector<Eigen::VectorXd> samples;
static std::vector<double> observations;

void kernelLFOpt(benchmark::State& state)
{
	// constexpr int numSamples = 800;
	constexpr int numSamples = 500;
	constexpr int dim = 10;

	if (samples.empty()) {
		for (int i = 0; i < numSamples; i++)
		{
			auto [s, o] = generateObservation(dim);
			samples.push_back(std::move(s));
			observations.push_back(o);
		}
	}

	double grad;
	switch (state.range(0))
	{
	case 0:
		grad = 0;
		break;
	case 1:
		grad = 1e-6;
		break;
	case 2:
		grad = 1e-3;
		break;
	default:
		throw std::runtime_error("E");
	}
	Params::opt_irpropplus::set_min_gradient(grad);
	Params::opt_rprop::set_eps_stop(grad);


	if (state.range(1) == 0) {
		limbo::model::GP<
			limbo::kernel::SquaredExpARD<Params::kernel_opt, Params::kernel_squared_exp_ard>,
			limbo::mean::Data,
			limbo::model::gp::KernelLFOpt<limbo::opt::Rprop<Params::opt_rprop>>
		> gp(dim);

		gp.initialize(samples, observations);
		for (auto _ : state)
		{
			gp.optimize_hyperparams();
			std::cout << gp.compute_log_lik() << "\n";
		}
	}
	else if (state.range(1) == 1){
		limbo::model::GP<
			limbo::kernel::SquaredExpARD<Params::kernel_opt, Params::kernel_squared_exp_ard>,
			limbo::mean::Data,
			limbo::model::gp::KernelLFOpt<limbo::opt::Irpropplus<Params::opt_irpropplus>>
		> gp(dim);

		gp.initialize(samples, observations);
		for (auto _ : state)
		{
			gp.optimize_hyperparams();
			std::cout << gp.compute_log_lik() << "\n";
		}
	}
	else if (state.range(1) == 2) {
		Params::opt_parallel::set_repeats(3);
		limbo::model::GP<
			limbo::kernel::SquaredExpARD<Params::kernel_opt, Params::kernel_squared_exp_ard>,
			limbo::mean::Data,
			limbo::model::gp::KernelLFOpt<limbo::opt::ParallelRepeater<Params::opt_parallel, limbo::opt::Irpropplus<Params::opt_irpropplus>>>
		> gp(dim);

		gp.initialize(samples, observations);
		for (auto _ : state)
		{
			gp.optimize_hyperparams();
			std::cout << gp.compute_log_lik() << "\n";
		}

	}
	else if (state.range(1) == 3) {
		Params::opt_parallel::set_repeats(2);
		limbo::model::GP<
			limbo::kernel::SquaredExpARD<Params::kernel_opt, Params::kernel_squared_exp_ard>,
			limbo::mean::Data,
			limbo::model::gp::KernelLFOpt<limbo::opt::ParallelRepeater<Params::opt_parallel, limbo::opt::Irpropplus<Params::opt_irpropplus>>>
		> gp(dim);

		gp.initialize(samples, observations);
		for (auto _ : state)
		{
			gp.optimize_hyperparams();
			std::cout << gp.compute_log_lik() << "\n";
		}

	}
	else 
	{
		throw std::runtime_error("E");
	}
	
}

// BENCHMARK(kernelLFOpt)->ArgsProduct({ {0}, {2, 3} })->Unit(benchmark::kMillisecond);
BENCHMARK(kernelLFOpt)->ArgsProduct({ {0, 1, 2}, {0, 1, 2, 3} })->Unit(benchmark::kMillisecond);
