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
#ifndef LIMBO_OPT_NLOPT_BASE_HPP
#define LIMBO_OPT_NLOPT_BASE_HPP

#ifndef USE_NLOPT
#warning No NLOpt
#else
#include <Eigen/Core>
#include <vector>
#include <nlopt.hpp>
#include <spdlog/spdlog.h>

namespace limbo::opt {
    /**
      @ingroup opt
    Base class for NLOpt wrappers
    */
    class NLOptBase {
    public:
        template <concepts::EvalFunc F>
        Eigen::VectorXd optimize(const F& f, const Eigen::VectorXd& init, std::optional<std::vector<std::pair<double, double>>> const& bounds, double* bestVal = nullptr) const
        {
            assert(init.size() == dim_);

            _opt.set_max_objective(nlopt_func<F>, (void*)&f);

            std::vector<double> x(dim_);
            Eigen::VectorXd::Map(&x[0], dim_) = init;

            if (bounds.has_value()) {
                std::vector<double> lowerBounds;
                std::vector<double> upperBounds;
                std::for_each(bounds.value().begin(), bounds.value().end(), [&lowerBounds, &upperBounds](auto const& bound) {lowerBounds.push_back(bound.first); upperBounds.push_back(bound.second); });
                _opt.set_lower_bounds(lowerBounds);
                _opt.set_upper_bounds(upperBounds);
            }

            double max;

            // try {
            //     nlopt::result result = _opt.optimize(x, max);
            // }
            // catch (nlopt::roundoff_limited const& e) {
            //     // In theory it's ok to ignore these errors
            //     spdlog::error("[NLOptNoGrad roundoff_limited]: {}", e.what());
            // }
            // catch (std::invalid_argument const& e) {
            //     // In theory it's ok to ignore this error
            //     spdlog::error("[NLOptNoGrad invalid_argument]: {}", e.what());
            // }
            // catch (std::runtime_error const& e) {
            //     // In theory it's ok to ignore this error
            //     spdlog::error("[NLOptGrad runtime_error]: {}", e.what());
            // }
            nlopt::result result = _opt.optimize(x, max);

            if (bestVal)
            {
                *bestVal = max;
            }


            return Eigen::VectorXd::Map(x.data(), x.size());
        }

        // Inequality constraints of the form f(x) <= 0
        template <concepts::EvalFunc F>
        void add_inequality_constraint(F* f, double tolerance = 1e-8)
        {
            if (!supportsInequalityConstraints(_opt.get_algorithm()))
            {
                throw std::runtime_error("This NLOPT algorithm does not support inequality constraints");
            }
            _opt.add_inequality_constraint(nlopt_func<F>, (void*)f, tolerance);
        }

        // Equality constraints of the form f(x) = 0
        template <concepts::EvalFunc F>
        void add_equality_constraint(F* f, double tolerance = 1e-8)
        {
            if (!supportsEqualityConstraints(_opt.get_algorithm()))
            {
                throw std::runtime_error("This NLOPT algorithm does not support equality constraints");
            }
            _opt.add_equality_constraint(nlopt_func<F>, (void*)f, tolerance);
		}

        nlopt::algorithm getAlgorithm() const { return _opt.get_algorithm(); }

    protected:

        NLOptBase(nlopt::algorithm algorithm, int dim) :
            dim_(dim)
        {
            _opt = nlopt::opt(algorithm, dim);
        }

        mutable nlopt::opt _opt;

    private:
        template <concepts::EvalFunc F>
        static double nlopt_func(const std::vector<double>& x, std::vector<double>& grad, void* my_func_data)
        {
            F* f = (F*)(my_func_data);
            Eigen::VectorXd params = Eigen::VectorXd::Map(x.data(), x.size());
            auto [funcVal, gradient] = (*f)(params, !grad.empty());

            if (!grad.empty()) { // Copy our new gradient data to nlopt's array.
                Eigen::VectorXd::Map(&grad[0], gradient.value().size()) = gradient.value();
            }
            return funcVal;
        }

        static bool supportsInequalityConstraints(nlopt::algorithm alg)
        {
	        switch (alg)
	        {
	        case nlopt::algorithm::AUGLAG:
	        case nlopt::algorithm::AUGLAG_EQ:
	        case nlopt::algorithm::GN_ISRES:
	        case nlopt::algorithm::GN_AGS:
	        case nlopt::algorithm::GN_ORIG_DIRECT:
	        case nlopt::algorithm::LN_COBYLA:
	        case nlopt::algorithm::LD_MMA:
	        case nlopt::algorithm::LD_SLSQP:
                return true;
	        default:
                return false;
	        }
        }

        static bool supportsEqualityConstraints(nlopt::algorithm alg)
        {
            switch (alg)
            {
            case nlopt::algorithm::AUGLAG:
            case nlopt::algorithm::AUGLAG_EQ:
            case nlopt::algorithm::GN_ISRES:
            case nlopt::algorithm::LN_COBYLA:
            case nlopt::algorithm::LD_SLSQP:
                return true;
            default:
                return false;
            }
        }

        int dim_;
    };
} // namespace limbo::opt

#endif
#endif
