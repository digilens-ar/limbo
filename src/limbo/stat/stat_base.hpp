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
#ifndef LIMBO_STAT_STAT_BASE_HPP
#define LIMBO_STAT_STAT_BASE_HPP

#include <fstream>
#include <string>

namespace limbo {
    namespace stat {
        /**
          Base class for statistics
        */
        struct StatBase {
            StatBase() {}

            template<typename T>
            static constexpr bool always_false = false;

            /// main method (to be written in derived classes)
            template <typename BO >
            void operator()(const BO& bo)
            {
                static_assert(always_false<BO>);
            }

            void setOutputDirectory(std::filesystem::path const& dir)
            {
                dir_ = dir;
            }

        protected:

            /**
             * return a reference to the log file for this stat. If a file for this stat does not already exist, the file will be created with `name`.
             * @param name 
             */
            std::ofstream& get_log_file(std::string const& name)
            {
                if (log_file_.has_value())
                    return *log_file_;

                char date[30];
                time_t date_time;
                time(&date_time);
                strftime(date, 30, "%Y-%m-%d_%H_%M_%S", localtime(&date_time));

                std::filesystem::path res_dir = dir_ / ("stats_" + std::string(date) + "_" + std::to_string(::_getpid()));
                if (!exists(res_dir))
                {
                    create_directory(res_dir);
                }
                std::filesystem::path log = res_dir / name;
                log_file_ = std::ofstream(log.c_str());
                assert(log_file_->good());
                return *log_file_;
            }

            std::filesystem::path const& get_log_directory() const
            {
                return dir_;
            }

        private:
            std::optional<std::ofstream> log_file_;
            std::filesystem::path dir_ = "";

        };
    }
}

#endif
