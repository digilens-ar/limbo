#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <random>
#include <limbo/limbo.hpp>

#include "testfunctions.hpp"

using TestFunc = Hartmann6;

static std::tuple<Eigen::VectorXd, double> generateObservation(size_t n)
{
	auto sample = Eigen::VectorXd::Random(TestFunc::dim_in(), 1);
	return { sample, TestFunc()(sample)};
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

enum GP_Type
{
	R_PROP,
	IR_PROP_PLUS,
	PARALLEL_RPT_2,
	PARALLEL_RPT_4,
};


template<GP_Type ModelType>
struct Model
{
	static_assert(false);
};

template<>
struct Model<R_PROP>
{
	using GP = limbo::model::GaussianProcess<
		limbo::kernel::SquaredExpARD<Params::kernel_opt, Params::kernel_squared_exp_ard>,
		limbo::mean::Data,
		limbo::model::gp::KernelLFOpt<limbo::opt::Rprop<Params::opt_rprop>>
	>;
};

template<>
struct Model<IR_PROP_PLUS>
{
	using GP = limbo::model::GaussianProcess<
		limbo::kernel::SquaredExpARD<Params::kernel_opt, Params::kernel_squared_exp_ard>,
		limbo::mean::Data,
		limbo::model::gp::KernelLFOpt<limbo::opt::Irpropplus<Params::opt_irpropplus>>
	>;
};

template<>
struct Model<PARALLEL_RPT_2>
{
	using GP = limbo::model::GaussianProcess<
		limbo::kernel::SquaredExpARD<Params::kernel_opt, Params::kernel_squared_exp_ard>,
		limbo::mean::Data,
		limbo::model::gp::KernelLFOpt<limbo::opt::ParallelRepeater<Params::opt_parallel, limbo::opt::Irpropplus<Params::opt_irpropplus>>>
	>;
};

template<>
struct Model<PARALLEL_RPT_4> : Model<PARALLEL_RPT_2>
{};


template<GP_Type ModelType>
void BM_KernelHPTune(benchmark::State& state)
{
	constexpr int numSamples = 1000;
	constexpr int dim = TestFunc::dim_in();

	if (samples.empty()) {
		samples.reserve(numSamples);
		observations.reserve(numSamples);
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
		grad = 1e-6;
		break;
	case 1:
		grad = 1e-3;
		break;
	case 2:
		grad = 1e-1;
		break;
	default:
		throw std::runtime_error("E");
	}
	Params::opt_irpropplus::set_min_gradient(grad);
	Params::opt_rprop::set_eps_stop(grad);

	if constexpr (ModelType == PARALLEL_RPT_2)
	{
		Params::opt_parallel::set_repeats(2);
	}
	else if constexpr (ModelType == PARALLEL_RPT_4)
	{
		Params::opt_parallel::set_repeats(4);
	}

	double before = 0;
	double after = 0;
	for (auto _ : state) {
		using GP_t = typename Model<ModelType>::GP;
		auto gp = GP_t::createFromSamples(samples, observations);
		before = gp.compute_log_lik();
		gp.optimize_hyperparams();
		after = gp.compute_log_lik();
	}
	std::cout << std::format("LogLik Before {}, After {}\n", before, after);
}

BENCHMARK(BM_KernelHPTune<R_PROP>)->ArgNames({"MinGradient"})->Arg(0)->Arg(1)->Arg(2)->Unit(benchmark::kMillisecond)->MinTime(6);
BENCHMARK(BM_KernelHPTune<IR_PROP_PLUS>)->ArgNames({"MinGradient"})->Arg(0)->Arg(1)->Arg(2)->Unit(benchmark::kMillisecond)->MinTime(6);
BENCHMARK(BM_KernelHPTune<PARALLEL_RPT_2>)->ArgNames({"MinGradient"})->Arg(0)->Arg(1)->Arg(2)->Unit(benchmark::kMillisecond)->MinTime(6);
BENCHMARK(BM_KernelHPTune<PARALLEL_RPT_4>)->ArgNames({"MinGradient"})->Arg(0)->Arg(1)->Arg(2)->Unit(benchmark::kMillisecond)->MinTime(6);
