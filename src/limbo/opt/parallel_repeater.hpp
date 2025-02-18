//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/limbo
//| Documentation: http://www.resibots.eu/limbo
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
#ifndef LIMBO_OPT_PARALLEL_REPEATER_HPP
#define LIMBO_OPT_PARALLEL_REPEATER_HPP

#include <algorithm>

#include <Eigen/Core>

#include <limbo/opt/optimizer.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/tools/parallel.hpp>
#include <limbo/tools/random.hpp>

namespace limbo {
    namespace defaults {
        struct opt_parallelrepeater {
            /// @ingroup opt_defaults
            /// number of replicates
            BO_PARAM(int, repeats, 10);

            /// epsilon of deviation: init + [-epsilon,epsilon]
            BO_PARAM(double, epsilon, 1e-2);
        };
    }
    namespace opt {
        /// @ingroup opt
        /// Meta-optimizer: run the same algorithm in parallel many times from different init points and return the maximum found among all the replicates
        /// (useful for local algorithms)
        ///
        /// Parameters:
        /// - int repeats
        template <typename opt_parallelrepeater, concepts::Optimizer Optimizer>
        struct ParallelRepeater {

            static ParallelRepeater create(int dims)
            {
                return ParallelRepeater();
            }

            template <concepts::EvalFunc F>
            Eigen::VectorXd optimize(F const& f, const Eigen::VectorXd& init, std::optional<std::vector<std::pair<double, double>>> const& bounds) const
            {
                assert(opt_parallelrepeater::repeats() > 0);
                assert(opt_parallelrepeater::epsilon() > 0.);
                using pair_t = std::pair<Eigen::VectorXd, double>;

                auto body = [&init, &bounds, &f](int i) {
                    Eigen::VectorXd newPoint;
                    if (i == 0)
                    { // For the first repeat don't do any randomization
                        newPoint = init;
                    }
                    else
                    {
                        Eigen::VectorXd r_deviation = tools::random_vector(init.size()).array() * 2. * opt_parallelrepeater::epsilon() - opt_parallelrepeater::epsilon();
                    	newPoint = init + r_deviation;
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

                auto comp = [](const pair_t& v1, const pair_t& v2) {
                    
                    return v1.second > v2.second;
                    
                };

                pair_t init_v = std::make_pair(init, -std::numeric_limits<float>::max());
                auto m = tools::par::max(init_v, opt_parallelrepeater::repeats(), body, comp);

                return m.first;
            };
        };
    }
}

#endif
