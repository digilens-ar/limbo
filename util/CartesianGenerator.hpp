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

        size_t totalIterations() const
        {
            return totalIterations_;
        }

        size_t currentIteration() const
        {
            return currentIteration_;
        }
    private:
        std::vector<unsigned> numPerDimension_;
        std::vector<unsigned> currentIdx_;
        unsigned currentDim_;
        size_t totalIterations_;
        size_t currentIteration_;
    };

}
