#pragma once
#include <concepts>

namespace limbo::concepts
{
	template <typename F, typename Ret, class... Args >
	concept Callable = std::invocable<F, Args...> && std::same_as<Ret, std::invoke_result_t<F, Args...>>;

	template <typename F>
	concept DoubleParam = Callable<F, double>;

	template <typename F>
	concept Option = Callable<F, bool>;

	// Param Types
	template<typename T>
	concept AcquiEI = requires (T a)
	{
		{ T::jitter() } -> std::convertible_to<double>;
	};

	//Fundamental Types
	template<typename T>
	concept Model = requires (T a)
	{
        { a.compute(std::vector<Eigen::VectorXd>{}, std::vector<Eigen::VectorXd>{}) } -> std::convertible_to<void>;
        { a.add_sample(Eigen::VectorXd{}, Eigen::VectorXd{}) } -> std::convertible_to<void>;
        { a.query(Eigen::VectorXd{}) } -> std::convertible_to<std::tuple<Eigen::VectorXd, double>>;
        { a.mu(Eigen::VectorXd{}) } -> std::convertible_to<Eigen::VectorXd>;
        { a.sigma(Eigen::VectorXd{}) } -> std::convertible_to<double>;
        { a.dim_in() } -> std::convertible_to<int>;
        { a.dim_out() } -> std::convertible_to<int>;
        { a.nb_samples() } -> std::convertible_to<int>;
        { a.samples() } -> std::convertible_to<const std::vector<Eigen::VectorXd>&>;
	};

	// Represents the objective function, takes in a coordinate of `dim_in` and returns a coordinate of `dim_out`.
	template <typename T>
	concept StateFunc = Callable<T, Eigen::VectorXd, Eigen::VectorXd>
	&&
	requires (T a)
	{
		{ a.dim_out() } -> std::convertible_to<size_t>;
		{ a.dim_in() } -> std::convertible_to<size_t>;
	};

	struct StateFuncArchetype
	{
		Eigen::VectorXd operator()(Eigen::VectorXd const&) { return Eigen::VectorXd(1); }
		size_t dim_out() const { return 1; }
		size_t dim_in() const { return 1; }
	};


	// This function is responsible for taking in the multidimensional observations and converting to a scalar `score` of the objective at that observation.
	template <typename T>
	concept AggregatorFunc = Callable<T, double, Eigen::VectorXd>;

	struct AggregatorFuncArchetype
	{
		double operator()(Eigen::VectorXd const& in) { return 0.0; }
	};

	template<typename T>
	concept BayesOptimizer = requires (T a)
	{
		{a.eval_and_add(StateFuncArchetype{}, Eigen::VectorXd()) } -> std::convertible_to<void>;
	};

	struct BayesOptimizerArchetype
	{
		void eval_and_add(StateFuncArchetype stateFunc, Eigen::VectorXd param) {};
	};

	template <typename T>
	concept StoppingCriteria = requires (T a)
	{
		{ a.operator()(BayesOptimizerArchetype(), AggregatorFuncArchetype()) } -> std::convertible_to<bool>;
	};

	// An evaluationFunction takes a coordinate and a t/f if gradient needs to be calculated and returns the objective function value and optionally the gradient at that location.
	template<typename T> // TODO this seems to be duplicate with AcquisitionFunc, just slightly different. In practice boptimizer is wrapping eval func around acquisitionfunc
	concept EvalFunc = Callable<T, std::pair<double, std::optional<Eigen::VectorXd>>, Eigen::VectorXd, bool>;

	struct EvalFuncArchetype
	{
		std::pair<double, std::optional<Eigen::VectorXd>> operator()(Eigen::VectorXd, bool) { return std::pair { 0.0, std::nullopt }; }
	};

	// An acquisition function acts as the objective function of the bayesian surrogate model. It is optimized upon to find the next point to evaluate the true objective function at.
	template <typename T>
	concept AcquisitionFunc = requires (T a)
	{
		{ a.operator()(Eigen::VectorXd{}, AggregatorFuncArchetype{}, true) } -> std::convertible_to<std::pair<double, std::optional<Eigen::VectorXd>>>;
	};

	/// An optimizer has an optimize method that takes an evalutionFunction, an initial coordinate, and a t/f if bounded, and returns a new optimum coordinate
	template<typename T>
	concept Optimizer = requires (T a)
	{
		{ T::create(3) } -> std::convertible_to<T>;
		{ a.optimize(EvalFuncArchetype{}, Eigen::VectorXd{}, true) } -> std::convertible_to<Eigen::VectorXd>;
	};

	//Receives data about the model and saves/logs it somewhere
	template <typename T>
	concept StatsFunc = requires (T a)
	{
		{ a.operator()(BayesOptimizerArchetype{}, AggregatorFuncArchetype{}) } -> std::convertible_to<void>;
	};
}