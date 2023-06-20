#pragma once
#include <concepts>
#include <Eigen/Dense>

namespace limbo
{
	enum EvaluationStatus
	{
		OK, // Return this if everything was successful
		SKIP, // return this if you want evaluation of the current point to be skipped.
		TERMINATE // return this to call for early termination of the optimization
	};
}

namespace limbo::concepts
{
	template <typename F, typename Ret, class... Args >
	concept Callable = std::invocable<F, Args...> && std::same_as<Ret, std::invoke_result_t<F, Args...>>;


	//Fundamental Types
	template<typename T>
	concept Model = requires (T a)
	{
        { a.initialize(std::vector<Eigen::VectorXd>{}, std::vector<double>{}) } -> std::convertible_to<void>;
        { a.add_sample(Eigen::VectorXd{}, double()) } -> std::convertible_to<void>;
        { a.query(Eigen::VectorXd{}) } -> std::convertible_to<std::tuple<double, double>>;
        { a.mu(Eigen::VectorXd{}) } -> std::convertible_to<double>;
        { a.sigma_sq(Eigen::VectorXd{}) } -> std::convertible_to<double>;
        { a.dim_in() } -> std::convertible_to<int>;
        { a.nb_samples() } -> std::convertible_to<int>;
        { a.samples() } -> std::convertible_to<const std::vector<Eigen::VectorXd>&>;
	};

	// Represents the objective function, takes in a coordinate of `dim_in`.
	template <typename T>
	concept StateFunc = Callable<T, std::tuple<EvaluationStatus, double>, Eigen::VectorXd>
	&&
	requires (T a)
	{
		{ a.dim_in() } -> std::convertible_to<size_t>;
	};

	struct StateFuncArchetype
	{
		std::tuple<EvaluationStatus,double> operator()(Eigen::VectorXd const&) { return { EvaluationStatus::OK, 1 }; }
		size_t dim_in() const { return 1; }
	};

	static_assert(StateFunc<StateFuncArchetype>);

	// An evaluationFunction takes a coordinate and a t/f if gradient needs to be calculated and returns the objective function value and optionally the gradient at that location.
	template<typename T>
	concept EvalFunc = Callable<T, std::pair<double, std::optional<Eigen::VectorXd>>, Eigen::VectorXd, bool>;

	struct EvalFuncArchetype
	{
		std::pair<double, std::optional<Eigen::VectorXd>> operator()(Eigen::VectorXd, bool) { return std::pair{ 0.0, std::nullopt }; }
	};

	template<typename T>
	concept BayesOptimizer = requires (T a)
	{
		{a.optimize(StateFuncArchetype{}, true) } -> std::convertible_to<std::string>;
		{a.eval_and_add(StateFuncArchetype{}, Eigen::VectorXd()) } -> std::convertible_to<EvaluationStatus>;
		{a.isBounded()} -> std::convertible_to<bool>;
		{a.addInequalityConstraint(EvalFuncArchetype{})} -> std::convertible_to<void>;
		{a.addEqualityConstraint(EvalFuncArchetype{})} -> std::convertible_to<void>;
		{a.constraintsAreSatisfied(Eigen::VectorXd{})} -> std::convertible_to<bool>;
		{a.hasConstraints()} -> std::convertible_to<bool>;
	};

	struct BayesOptimizerArchetype
	{
		EvaluationStatus eval_and_add(StateFuncArchetype stateFunc, Eigen::VectorXd param) { return OK; }
		std::string optimize(StateFuncArchetype stateFunc, bool reset) { return "Ended for no reason"; }
		bool isBounded() { return true; }
		void addInequalityConstraint(EvalFuncArchetype arch) {}
		void addEqualityConstraint(EvalFuncArchetype arch) {}
		bool constraintsAreSatisfied(Eigen::VectorXd sampleLoc) { return true; }
		bool hasConstraints() {return true;}
	};

	static_assert(BayesOptimizer<BayesOptimizerArchetype>);

	// A function which generates the initial samples to create the baysian model.
	template<typename T>
	concept InitFunc = requires (T a)
	{
		{ a.operator()(StateFuncArchetype{}, BayesOptimizerArchetype{}) } -> std::convertible_to<EvaluationStatus>;
	};

	template <typename T>
	concept StoppingCriteria = requires (T a, std::string& stoppingMessage)
	{
		{ a(BayesOptimizerArchetype(), stoppingMessage) } -> std::convertible_to<bool>;
	};

	// An acquisition function acts as the objective function of the bayesian surrogate model. It is optimized upon to find the next point to evaluate the true objective function at.
	template <typename T>
	concept AcquisitionFunc = requires (T a)
	{
		{ a.operator()(Eigen::VectorXd{}, true) } -> std::convertible_to<std::pair<double, std::optional<Eigen::VectorXd>>>;
	};

	/// An optimizer has an optimize method that takes an evalutionFunction, an initial coordinate, and an optional set of lower and upper bounds for the parameters, and returns a new optimum coordinate
	template<typename T>
	concept Optimizer = requires (T a)
	{
		{ T::create(3) } -> std::convertible_to<T>;
		{ a.optimize(EvalFuncArchetype{}, Eigen::VectorXd{}, std::optional<std::vector<std::pair<double, double>>>{}) } -> std::convertible_to<Eigen::VectorXd>;
	};

	//Receives data about the model and saves/logs it somewhere
	template <typename T>
	concept StatsFunc = requires (T a)
	{
		{ a.operator()(BayesOptimizerArchetype{}) } -> std::convertible_to<void>;
	} &&
		std::is_default_constructible_v<T>;
}