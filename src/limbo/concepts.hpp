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

	template<typename T>
	concept AcquiEI = requires (T a)
	{
		{ T::jitter() } -> std::convertible_to<double>;
	};

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
}