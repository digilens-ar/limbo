#pragma once
#include <vector>


namespace limbo::serialize
{

    class CartesianGenerator
    {
    public:
        CartesianGenerator(std::vector<unsigned> numPerDimension);

        std::vector<unsigned> const& currentIndex() const
        {
            return currentIdx_;
        }

        void iterate();

        unsigned totalIterations() const
        {
            return totalIterations_;
        }

        unsigned currentIteration() const
        {
            return currentIteration_;
        }
    private:
        std::vector<unsigned> numPerDimension_;
        std::vector<unsigned> currentIdx_;
        unsigned currentDim_;
        unsigned totalIterations_;
        unsigned currentIteration_;
    };

}
