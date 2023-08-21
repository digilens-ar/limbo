#pragma once

#include <limbo/tools/macros.hpp>
#include <spdlog/fmt/fmt.h>

namespace limbo {
    namespace defaults {
        struct stop_max_merit_value {
            /// @ingroup stop_defaults
            BO_PARAM(double, stopValue, 1);
            BO_PARAM(bool, enabled, true);
        };
    }
    namespace stop {
        /// @ingroup stop
        /// Stop after reaching a merit function value.
        ///
        /// parameter: double stopValue
        /// parameter: bool enabled
        template <typename stop_max_merit_value>
        struct MaxMeritValue {

            template <typename BO>
            bool operator()(const BO& bo, std::string& stopMessage) const
            {
                if (!stop_max_merit_value::enabled())
                    return false;
                if (bo.best_observation() >= stop_max_merit_value::stopValue())
                {
                    stopMessage = fmt::format("The maximum merit function value of {} was reached", stop_max_merit_value::stopValue());
                    return true;
                }
                else
                {
                    return false;
                }
            }
        };
    }
}