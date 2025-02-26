#include "CartesianGenerator.hpp"

namespace limbo::serialize {
	CartesianGenerator::CartesianGenerator(std::vector<unsigned> numPerDimension):
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

	void CartesianGenerator::iterate()
	{
		if (currentIdx_.at(currentDim_) == numPerDimension_.at(currentDim_) - 1)
		{ // we are at the maximum value for the current axis. Need to set value back to 0 and switch to a different axis.
			currentIdx_.at(currentDim_) = 0;
			if (currentDim_ == 0)
			{ // this should only happen once we've iterated through the whole thing and need to roll back to 00000
				currentDim_ = static_cast<unsigned>(numPerDimension_.size()) - 1; // reset back to first iteration
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
}
