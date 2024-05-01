#pragma once

// Public types without including heavy implementation
namespace limbo
{
	enum EvaluationStatus
	{
		OK, // Return this if everything was successful
		SKIP, // return this if you want evaluation of the current point to be skipped.
		TERMINATE // return this to call for early termination of the optimization
	};

	
    class EvaluationError : public std::exception
    {
    public:
        using std::exception::exception;
    };
}