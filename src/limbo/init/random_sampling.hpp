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
#ifndef LIMBO_INIT_RANDOM_SAMPLING_HPP
#define LIMBO_INIT_RANDOM_SAMPLING_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>
#include <limbo/tools/random.hpp>
#include <limbo/concepts.hpp>
#include <spdlog/spdlog.h>

namespace limbo {
    namespace defaults {
        struct init_randomsampling {
            ///@ingroup init_defaults
            BO_PARAM(int, samples, 10);
        };
    }
    namespace init {
        /** @ingroup init
          \rst
          Pure random sampling in [0, 1]^n

          Parameters:
            - ``int samples`` (total number of samples)
          \endrst
        */
        template <typename InitRandomSampling>
        struct RandomSampling {
            template <concepts::StateFunc StateFunction, concepts::BayesOptimizer Opt>
            EvaluationStatus operator()(const StateFunction& seval,  Opt& opt) const
            {
                for (int i = 0; i < InitRandomSampling::samples(); i++) {
                    Eigen::VectorXd new_sample;
                    if (opt.hasConstraints())
                    {
                        const auto proposed_new_sample = tools::random_vector(seval.dim_in(), opt.isBounded());
                        auto ConstFunc = [&proposed_new_sample](Eigen::VectorXd const& position, bool gradient) -> std::pair<double, std::optional<Eigen::VectorXd>> // A function with a max at proposed_new_sample that falls off proportional to the distance.
                        { 
                            assert(!gradient);
                            return {
                                -.01 * (proposed_new_sample - position).norm(),
                                std::nullopt};
                        };
                        // static_assert(concepts::EvalFunc<ConstFunc>);

                        std::optional<std::vector<std::pair<double, double>>> parameterBounds = std::nullopt;
                        if (opt.isBounded()) {
                            parameterBounds = std::vector<std::pair<double, double>>(seval.dim_in(), std::make_pair(0.0, 1.0));
                        }

                        //find the closest coordinate to proposed_new_sample that satisfies the constraints
	                    new_sample = opt.acquisition_optimizer().optimize(
                            ConstFunc, 
                            proposed_new_sample, 
                            parameterBounds);
                    }
                    else
                    {
	                    new_sample = tools::random_vector(seval.dim_in(), opt.isBounded());
                    }

					EvaluationStatus status = opt.eval_and_add(seval, new_sample);
                    assert(status != SKIP); // I'm not sure how we should handle this case.

					if (status == TERMINATE)
					{
                        return TERMINATE;
					}
                }
                return OK;
            }
        };


        //This initialization routine extends `RandomSampling` with an initial measurement at an externally provided coordinate
    	template<typename InitRandomSampling>
        struct RandomSamplingWithSingleInit : RandomSampling<InitRandomSampling>
    	{
            void setInitialPoint(std::optional<Eigen::VectorXd> const& init)
            {
                initialPoint_ = init;
            }

            template <concepts::StateFunc StateFunction, concepts::BayesOptimizer Opt>
            EvaluationStatus operator()(const StateFunction& seval, Opt& opt) const
            {
                if (initialPoint_.has_value())
                {
                    EvaluationStatus status = opt.eval_and_add(seval, initialPoint_.value());
                    assert(status != SKIP); // I'm not sure how we should handle this case.
                    if (status == TERMINATE)
                    {
                        return TERMINATE;
                    }
                }
                return RandomSampling<InitRandomSampling>::operator()(seval, opt);
            }

            std::optional<Eigen::VectorXd> initialPoint_ = std::nullopt;
    	};

    }
}

#endif
