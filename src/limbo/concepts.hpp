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

	template <typename T>
	concept AggregatorFunc = Callable<T, double, Eigen::VectorXd>;

	template<typename T>
	concept BayesOptimizer = requires (T a)
	{
		{a.eval_and_add(StateFuncArchetype{}, Eigen::VectorXd()) } -> std::convertible_to<void>;
	};

	//An "Evaluation function" takes a coordinate and a true/false of whether to calculate the gradient as input. Returns the value and gradient at that coordinate.
	template<typename T>
	concept EvalFunc = Callable<T, std::pair<double, std::optional<Eigen::VectorXd>>, Eigen::VectorXd, bool>;

	struct EvalFuncArchetype
	{
		std::pair<double, std::optional<Eigen::VectorXd>> operator()(Eigen::VectorXd, bool) { return std::pair { 0.0, std::nullopt }; }
	};

	template<typename T>
	concept Optimizer = requires (T a)
	{
		{ T::create(3) } -> std::convertible_to<T>;
		{ a.optimize(EvalFuncArchetype{}, Eigen::VectorXd{}, true) } -> std::convertible_to<Eigen::VectorXd>;
	};
}