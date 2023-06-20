#pragma once

#include <algorithm>

#include <Eigen/Core>

#include <limbo/opt/optimizer.hpp>
#include <limbo/tools/parallel.hpp>
#include <limbo/tools/random_generator.hpp>

namespace limbo {

    namespace opt {
        /// @ingroup opt
        /// Meta-optimizer: run the same algorithm in parallel twice. Once from the requested initial point and once from the first requested initial point. Return the best of the two results
        /// (useful for local algorithms)
        template <concepts::Optimizer Optimizer>
        struct ParallelRepeater2 {

            static ParallelRepeater2 create(int dims)
            {
                return ParallelRepeater2();
            }

            template <concepts::EvalFunc F>
            Eigen::VectorXd optimize(F const& f, const Eigen::VectorXd& init, std::optional<std::vector<std::pair<double, double>>> const& bounds) const
            {
          
                using pair_t = std::pair<Eigen::VectorXd, double>;

                auto body = [this, &init, &bounds, &f](int i) -> pair_t {
                    Eigen::VectorXd newPoint;
                    if (i == 0)
                    { 
                        newPoint = init;
                    }
                    else
                    {
                        newPoint = firstInit_.value();
                    }
                 
                    if (bounds.has_value())
                    { // Make sure initil point is in bounds
	                    for (int j=0; j<newPoint.size(); j++)
	                    {
                            newPoint(j) = std::clamp(newPoint(j), bounds.value().at(j).first, bounds.value().at(j).second);
	                    }
                    }
                    Eigen::VectorXd v = Optimizer::create(init.size()).optimize(f, newPoint, bounds);
                    double val = opt::eval(f, v);
                    return std::make_pair(v, val);
                };

                if (!firstInit_.has_value())
                { // If this is the first call then store the init point and do just a single run. On subsequent calls we will run init and firstInit_
                    firstInit_ = init;
                    return body(0).first;
                }

                auto comp = [](const pair_t& v1, const pair_t& v2) {
                    
                    return v1.second > v2.second;
                    
                };

                pair_t init_v = std::make_pair(init, -std::numeric_limits<float>::max());
                auto m = tools::par::max(init_v, 2, body, comp);

                return m.first;
            };

        private:
            mutable std::optional<Eigen::VectorXd> firstInit_ = std::nullopt;
        };
    }
}
